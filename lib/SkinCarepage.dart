import 'package:flutter/material.dart';
import 'package:tflite/tflite.dart';
import 'package:camera/camera.dart'; // camera stream
import 'dart:convert'; // JSON
import 'dart:math' as math;
import 'package:permission_handler/permission_handler.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'detection_state.dart';

enum ScanMode { typeOnly, problemOnly }

class SkinCarePage extends StatefulWidget {
  const SkinCarePage({super.key});

  @override
  _SkinCarePageState createState() => _SkinCarePageState();
}

class _SkinCarePageState extends State<SkinCarePage> {
  static const _typeModel = "assets/models/final_skin_model.tflite";
  static const _typeLabels = "assets/models/skin_type_labels.txt";
  static const _problemModel = "assets/models/final-skinproblem_model.tflite";
  static const _problemLabels = "assets/models/skin_problem_labels.txt";

  CameraController? _controller;
  bool _isRunning = false;
  bool _permissionDenied = false;
  bool _initializing = false;
  String? _initError;
  String _lastPrediction = "Awaiting camera...";
  bool _detectProblem = false; // problem model flag
  bool _modelLoading = false;
  ScanMode _scanMode = ScanMode.typeOnly; // user-selectable
  int _frameCount = 0;
  int _framesProcessed = 0;
  bool _modelReady = false;
  int _frameLogEvery = 15;
  bool _cameraReady = false;
  int _debugFrameLogLimit = 20;
  String _debugInfo = "";
  int _framesSeen = 0;
  int _inferenceAttempts = 0;
  String _cameraInfo = "";
  String _loadedModelKey = "";
  String? _detectedSkinType;
  String? _detectedSkinProblem;
  bool _pendingProblemScan = false; // used to show problem scanning state
  Map<String, dynamic>? _routineData;
  String _routineText = "Routine will appear here after detection.";
  double _minConfidenceType = 0.4; // further relaxed to accept outputs
  double _minConfidenceProblem = 0.0; // accept any confidence for problems
  double _minThresholdType = 0.5;
  double _minThresholdProblem = 0.0; // keep zero to avoid suppressing outputs
  int _stableRequiredType = 2;
  int _stableRequiredProblem = 1;
  int _stableCount = 0;
  String? _lastRawLabel;
  bool _problemDetected = false;
  bool _hasFace = false;
  int _problemAttempts = 0;
  int _problemAttemptsBeforeRelax = 6;
  int _typeAttempts = 0;
  int _typeAttemptsBeforeFallback = 6;
  bool _problemRelaxed = false;
  int _problemEmptyCount = 0;
  int _problemEmptyMax = 5;
  final Set<String> _knownTypes =
      {"normal", "dry", "combination", "oily"};
  final Set<String> _knownProblems = {
    "blackhead",
    "dark_circles",
    "pigmentation",
    "pimples",
    "wrinkles",
    "normal"
  };
  final double _minLuma = 40;
  final double _maxLuma = 210;
  final double _minLumaStd = 12;

  @override
  void initState() {
    super.initState();
    _ensureCameraPermission();
    _applyMode(_scanMode, initial: true);
    _loadRoutineData();
  }

  void _applyMode(ScanMode mode, {bool initial = false}) async {
    setState(() {
      _scanMode = mode;
      _detectProblem = mode == ScanMode.problemOnly;
      _pendingProblemScan = mode == ScanMode.problemOnly;
      _problemDetected = false;
      if (mode == ScanMode.problemOnly) {
        // Keep the detected skin type so the user can see it while running problem detection.
        _detectedSkinProblem = null;
      } else {
        // Type-only mode; keep any prior problem result but stop problem detection.
        _detectProblem = false;
        _pendingProblemScan = false;
      }
      _typeAttempts = 0;
      _problemAttempts = 0;
      _routineText = _routineText.isEmpty ? "Routine will appear here after detection." : _routineText;
      _debugInfo = "Mode: ${mode == ScanMode.problemOnly ? "Problem only" : "Type only"}";
      _lastPrediction = mode == ScanMode.problemOnly
          ? "Model ready: scanning for skin problems..."
          : "Model ready: scanning for skin type...";
      _modelReady = false;
      _loadedModelKey = "";
    });
    if (!initial) {
      await _loadModel();
    } else {
      _loadModel();
    }
  }

  String _composePredictionText() {
    final typeText = _detectedSkinType != null ? "Skin type: $_detectedSkinType" : null;
    String? problemText;
    if (_detectedSkinProblem != null) {
      problemText = "Skin problem: $_detectedSkinProblem";
    } else if (_scanMode == ScanMode.problemOnly && (_pendingProblemScan || _detectProblem)) {
      problemText = "Scanning skin problems...";
    }
    if (typeText != null && problemText != null) {
      return "$typeText | $problemText";
    }
    return typeText ?? problemText ?? _lastPrediction;
  }

  Future<void> _loadRoutineData() async {
    try {
      final jsonStr =
          await rootBundle.loadString('assets/models/skin_routine_data.json');
      final data = json.decode(jsonStr) as Map<String, dynamic>;
      setState(() {
        _routineData = data;
      });
    } catch (e) {
      setState(() {
        _routineData = null;
        _routineText = "Failed to load routine data: $e";
      });
    }
  }

  String _normalizeLabel(String raw) {
    var label = raw.trim().toLowerCase();
    label = label.replaceAll(RegExp('^[0-9]+\\s*'), '');
    label = label.replaceAll(' ', '_');
    return label;
  }

  String _formatLabel(String normalized) {
    if (normalized.isEmpty) return "Unknown";
    return normalized
        .split('_')
        .where((part) => part.isNotEmpty)
        .map((part) => part[0].toUpperCase() + part.substring(1))
        .join(' ');
  }

  bool _looksLikeFace(CameraImage image) {
    if (image.planes.isEmpty) return false;
    final yPlane = image.planes.first.bytes;
    final w = image.width;
    final h = image.height;
    if (w == 0 || h == 0 || yPlane.isEmpty) return false;

    final regionW = (w / 4).clamp(20, 200).toInt();
    final regionH = (h / 4).clamp(20, 200).toInt();
    final startX = (w - regionW) ~/ 2;
    final startY = (h - regionH) ~/ 2;
    final stepX = (regionW / 16).clamp(1, 8).toInt();
    final stepY = (regionH / 16).clamp(1, 8).toInt();

    int count = 0;
    double sum = 0;
    double sumSq = 0;
    for (int y = startY; y < startY + regionH; y += stepY) {
      for (int x = startX; x < startX + regionW; x += stepX) {
        final idx = y * w + x;
        if (idx < 0 || idx >= yPlane.length) continue;
        final v = yPlane[idx].toDouble();
        sum += v;
        sumSq += v * v;
        count++;
      }
    }
    if (count == 0) return false;
    final mean = sum / count;
    final variance = (sumSq / count) - (mean * mean);
    final std = variance <= 0 ? 0 : math.sqrt(variance);
    return mean >= _minLuma && mean <= _maxLuma && std >= _minLumaStd;
  }

  void _updateRoutineText() {
    if (_routineData == null || _detectedSkinType == null) return;
    final typeKey = _normalizeLabel(_detectedSkinType!);
    final problemKey =
        _detectedSkinProblem != null ? _normalizeLabel(_detectedSkinProblem!) : null;

    final typeEntry = (_routineData?['skin_types'] ?? {})[typeKey];
    final problemEntry =
        problemKey != null ? (_routineData?['skin_problems'] ?? {})[problemKey] : null;

    final buffer = StringBuffer();
    buffer.writeln("Suggested routine (combined):");
    final amParts = <String>[];
    final pmParts = <String>[];
    final products = <String>[];

    if (typeEntry != null) {
      final routine = typeEntry['routine'] as Map<String, dynamic>?;
      amParts.add(routine?['AM'] ?? '');
      pmParts.add(routine?['PM'] ?? '');
      final kp = routine?['key_products'];
      if (kp != null) {
        products.add(kp.toString());
      }
    }
    if (problemEntry != null) {
      final routine = problemEntry['routine'] as Map<String, dynamic>?;
      amParts.add(routine?['AM'] ?? '');
      pmParts.add(routine?['PM'] ?? '');
      final kp = routine?['key_products'];
      if (kp != null) {
        products.add(kp.toString());
      }
    }

    final am = amParts.where((s) => s.trim().isNotEmpty).join(" + ");
    final pm = pmParts.where((s) => s.trim().isNotEmpty).join(" + ");
    if (am.isNotEmpty) buffer.writeln("AM: $am");
    if (pm.isNotEmpty) buffer.writeln("PM: $pm");
    if (products.isNotEmpty) {
      buffer.writeln("Key products: ${products.join(" | ")}");
    }

    final combined = buffer.isEmpty ? "No routine found for detected type/problem." : buffer.toString();
    setState(() {
      _routineText = combined;
    });
    if (_detectedSkinType != null && _detectedSkinProblem != null) {
      DetectionState.updateRoutine(combined);
    }
  }

  // Load the selected TFLite model (skin type/problem).
  Future<void> _loadModel() async {
    final targetKey = _detectProblem ? "problem" : "type";
    if (_modelLoading) {
      setState(() {
        _debugInfo = "Model already loading...";
      });
      return;
    }
    if (_modelReady && _loadedModelKey == targetKey) {
      setState(() {
        _debugInfo = "Model ready";
      });
      return;
    }
    _modelLoading = true;
    try {
      if (_loadedModelKey.isNotEmpty && _loadedModelKey != targetKey) {
        await Tflite.close(); // close only when switching models
      }
      final res = await Tflite
          .loadModel(
        model: _detectProblem ? _problemModel : _typeModel,
        labels: _detectProblem ? _problemLabels : _typeLabels,
      )
          .timeout(const Duration(seconds: 8), onTimeout: () => null);
      setState(() {
        _lastPrediction = _detectProblem
            ? "Detecting skin problem..."
            : "Detecting skin type...";
      });
      _modelReady = res != null;
      if (_modelReady) {
        _loadedModelKey = targetKey;
      }
      setState(() {
        _lastPrediction = _modelReady
            ? (_detectProblem
                ? "Model ready: scanning for skin problems..."
                : "Model ready: scanning for skin type...")
            : "Failed to load model";
        if (_modelReady && _debugInfo.isEmpty) {
          _debugInfo = "Model ready";
        } else if (!_modelReady && _debugInfo.isEmpty) {
          _debugInfo = "Model load timeout or failure";
        }
      });
      print("TFLite loadModel result: $res");
      _framesProcessed = 0;
      _inferenceAttempts = 0;
    } catch (e) {
      setState(() {
        _lastPrediction = "Model load error: $e";
        _modelReady = false;
        _debugInfo = "Model load error: $e";
      });
    } finally {
      _modelLoading = false;
    }
  }

  // Camera permission and init
  Future<void> _ensureCameraPermission() async {
    setState(() {
      _initializing = true;
      _initError = null;
    });
    final status = await Permission.camera.request();
    if (!mounted) return;

    if (status.isGranted || status.isLimited) {
      setState(() {
        _permissionDenied = false;
      });
      await _initializeCamera();
    } else {
      setState(() {
        _permissionDenied = true;
        _initializing = false;
      });
    }
  }

  Future<void> _initializeCamera() async {
    try {
      final cameras = await availableCameras();
      if (cameras.isEmpty) {
        setState(() {
          _initError = "No camera available on this device.";
          _initializing = false;
        });
        return;
      }
      final camera = cameras.first;

      _controller = CameraController(
        camera,
        ResolutionPreset.medium,
        enableAudio: false,
      );

      await _controller!.initialize();
      if (!mounted) return;

      // Start streaming frames immediately; model may still be loading.
      await _controller!.startImageStream((CameraImage image) {
        _runOnFrame(image);
      });

      setState(() {
        _initializing = false;
        _cameraReady = true;
        _framesSeen = 0;
        _cameraInfo =
            "Lens: ${camera.lensDirection}, orientation: ${_controller!.description.sensorOrientation}";
        final readyMsg = _detectProblem
            ? "Camera ready. Hold steady to detect skin problems..."
            : "Camera ready. Hold steady to detect skin type...";
        _lastPrediction = _modelReady
            ? readyMsg
            : (_modelLoading
                ? "Camera ready. Loading model..."
                : "Camera ready. Model not ready.");
        if (_debugInfo.isEmpty) {
          _debugInfo = _modelReady
              ? "Model ready"
              : (_modelLoading ? "Loading model..." : "Model not ready");
        }
      });
    } catch (e) {
      setState(() {
        _initError = "Failed to start camera: $e";
        _initializing = false;
      });
    }
  }

  // Run inference on frames
  Future<void> _runOnFrame(CameraImage image) async {
    _framesSeen++;
    if (_framesSeen % _frameLogEvery == 0) {
      print("Camera stream alive; frames seen: $_framesSeen");
    }

    if (_isRunning || _modelLoading || !_modelReady) {
      if (_framesSeen % _frameLogEvery == 0 && mounted) {
        setState(() {
          _debugInfo =
              _modelReady ? "Skipping frames (busy)" : "Model still loading...";
        });
      }
      return;
    }
    // Keep face gate strict for skin type; be lenient for problem-only mode.
    final requireFaceCheck = !_detectProblem;
    if (requireFaceCheck && !_looksLikeFace(image)) {
      setState(() {
        _lastPrediction = "No face detected. Please center your face in the frame.";
        _debugInfo = "Heuristic: low/flat luminance, skipping.";
        if (!_detectProblem) {
          _detectedSkinType = null;
        }
        _detectedSkinProblem = null;
        _pendingProblemScan = false;
        _problemDetected = false;
        _typeAttempts = 0;
        _routineText = "Routine will appear here after detection.";
      });
      _stableCount = 0;
      _lastRawLabel = null;
      _hasFace = false;
      return;
    }

    _frameCount = (_frameCount + 1) % 30;

    // If skin type is already determined and we are still on type model, skip further type inference.
    if (_scanMode == ScanMode.typeOnly && _detectedSkinType != null) {
      return;
    }
    // If problem already determined, skip further problem inference.
    if (_detectProblem && _problemDetected) {
      return;
    }

    _isRunning = true;
    _inferenceAttempts++;

    try {
      if (_frameCount == 0 && mounted) {
        setState(() {
          _lastPrediction = "Processing frame...";
        });
      }
      if (_framesProcessed % _frameLogEvery == 0) {
        print(
            "Running inference; processed $_framesProcessed frames so far. modelReady=$_modelReady, frame $_frameCount");
      }
      if (_framesProcessed < _debugFrameLogLimit) {
        print(
            "Frame info -> planes: ${image.planes.length}, size: ${image.width}x${image.height}, rotation: ${_controller?.description.sensorOrientation}");
      }
      final rotation = _controller?.description.sensorOrientation ?? 0;
      final thresholdMain =
          _detectProblem ? _minThresholdProblem : _minThresholdType;
      List? output = await Tflite.runModelOnFrame(
        bytesList: image.planes.map((p) => p.bytes).toList(),
        imageHeight: image.height,
        imageWidth: image.width,
        numResults: 1,
        threshold: thresholdMain,
        imageMean: 127.5,
        imageStd: 127.5,
        rotation: rotation,
        asynch: false,
      );

      if (output == null || output.isEmpty) {
        final thresholdFallback =
            _detectProblem ? _minThresholdProblem : _minThresholdType;
        output = await Tflite.runModelOnFrame(
          bytesList: image.planes.map((p) => p.bytes).toList(),
          imageHeight: image.height,
          imageWidth: image.width,
          numResults: 1,
          threshold: thresholdFallback,
          imageMean: 0.0,
          imageStd: 255.0,
          rotation: (rotation + 90) % 360,
          asynch: false,
        );
        if (output == null || output.isEmpty) {
          output = [];
        } else {
          print("Fallback inference produced output: ${output.first}");
        }
      }

      if (output != null && output.isNotEmpty) {
        // Pick the entry with highest confidence (or confidenceInClass).
        final top = output.reduce((a, b) {
          final ca = (a['confidence'] ?? a['confidenceInClass'] ?? 0);
          final cb = (b['confidence'] ?? b['confidenceInClass'] ?? 0);
          final na = ca is num ? ca : 0;
          final nb = cb is num ? cb : 0;
          return nb > na ? b : a;
        });
        final rawLabel = top['label']?.toString() ?? 'Unknown';
        final normLabel = _normalizeLabel(rawLabel);
        final displayLabel = _formatLabel(normLabel);
        final confAny = top['confidence'] ?? top['confidenceInClass'] ?? 0;
        final conf = confAny is num ? confAny.toDouble() : 0.0;
        final isProblem = _detectProblem;
        if (isProblem) {
          print("Problem model output: $top");
        }
        if (isProblem) {
          _problemAttempts++;
          if (_problemAttempts > _problemAttemptsBeforeRelax) {
            _problemRelaxed = true;
          }
        }
        final minConf = isProblem ? _minConfidenceProblem : _minConfidenceType;
        final neededStable = isProblem ? _stableRequiredProblem : _stableRequiredType;

        // If the best confidence is below threshold, treat as no update.
        final badLabel = !isProblem && (normLabel.isEmpty || normLabel == "unknown");
        final reject = conf < minConf || badLabel;
        if (reject) {
          setState(() {
            _debugInfo =
                "Low/unknown output ${(conf * 100).toStringAsFixed(1)}% or label '$normLabel' not allowed.";
            if (isProblem) {
              _lastPrediction = _composePredictionText();
              _detectedSkinProblem = null;
              _problemDetected = false;
              _pendingProblemScan = true;
              print("Problem reject -> conf=${(conf * 100).toStringAsFixed(1)} label=$normLabel");
            } else {
              _typeAttempts++;
              _lastPrediction =
                  "No face detected. Please center your face in the frame.";
              _detectedSkinType = null;
              _pendingProblemScan = false;
              _routineText = "Routine will appear here after detection.";
            }
          });
          if (!isProblem && _typeAttempts >= _typeAttemptsBeforeFallback) {
            setState(() {
              _detectedSkinType = "Unknown";
              _pendingProblemScan = false;
              _lastPrediction = _composePredictionText();
              _debugInfo = "Type fallback after $_typeAttempts attempts";
            });
            _typeAttempts = 0;
          }
          _stableCount = 0;
          _lastRawLabel = null;
          if (!isProblem) {
            _hasFace = false;
          }
          _isRunning = false;
          return;
        }

        if (_lastRawLabel == rawLabel) {
          _stableCount++;
        } else {
          _stableCount = 1;
          _lastRawLabel = rawLabel;
        }

        if (_stableCount < neededStable) {
          setState(() {
            _lastPrediction = "Hold steady, detecting...";
            _debugInfo = "Stabilizing... ($displayLabel)";
          });
          _isRunning = false;
          return;
        }

        if (isProblem) {
          _detectedSkinProblem = displayLabel;
          _problemDetected = true;
          _pendingProblemScan = false;
          _problemRelaxed = false;
          _problemAttempts = 0;
          _problemEmptyCount = 0;
          _debugInfo = "Problem detected: $displayLabel @ ${(conf * 100).toStringAsFixed(1)}%";
        } else {
          _detectedSkinType = displayLabel;
          _typeAttempts = 0;
          _pendingProblemScan = false;
          _problemRelaxed = false;
          _problemAttempts = 0;
          _problemEmptyCount = 0;
        }
        setState(() {
          _lastPrediction = _composePredictionText();
          _debugInfo = "Top: $top";
        });
        _updateRoutineText();
        print("TFLite top result: $top");
        _framesProcessed++;
        _stableCount = 0;
        _lastRawLabel = null;
        _hasFace = true;
      } else {
        setState(() {
          if (_detectProblem) {
            _problemEmptyCount++;
            if (_problemEmptyCount >= _problemEmptyMax) {
              _detectedSkinProblem = "No problem detected";
              _problemDetected = true;
              _pendingProblemScan = false;
              _lastPrediction = _composePredictionText();
              _debugInfo = "No problem output; defaulted after $_problemEmptyCount tries";
            } else {
              _lastPrediction = _composePredictionText();
              _debugInfo = "Empty output (problem scan)";
              _detectedSkinProblem = null;
              _problemDetected = false;
              _pendingProblemScan = true;
              print("Problem model returned empty output (attempt $_problemEmptyCount)");
            }
          } else {
            _typeAttempts++;
            _lastPrediction =
                "No face detected. Please center your face in the frame.";
            _debugInfo = "Empty output";
            _detectedSkinType = null;
            _detectedSkinProblem = null;
            _pendingProblemScan = false;
            _problemDetected = false;
            _routineText = "Routine will appear here after detection.";
            if (_typeAttempts >= _typeAttemptsBeforeFallback) {
              _detectedSkinType = "Unknown";
              _pendingProblemScan = false;
              _debugInfo = "Type fallback after $_typeAttempts attempts (empty output)";
              _typeAttempts = 0;
            }
          }
        });
        _stableCount = 0;
        _lastRawLabel = null;
        _hasFace = false;
        print("TFLite returned empty output (both attempts)");
        if (_inferenceAttempts > 5 && _framesProcessed == 0 && mounted) {
          setState(() {
            _debugInfo =
                "No predictions yet; check lighting/face position. Try rotating device.";
          });
        }
        _isRunning = false;
        return;
      }

      // No auto-switch; user chooses mode manually.
    } catch (e) {
      setState(() {
        _lastPrediction = "TFLite error: $e";
        _debugInfo = "Error: $e";
      });
    } finally {
      _isRunning = false;
    }
  }

  @override
  void dispose() {
    _controller?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      extendBodyBehindAppBar: true,
      appBar: null,
      body: SafeArea(
        child: _permissionDenied
            ? Center(
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    const Text('Camera permission is required to continue.'),
                    const SizedBox(height: 12),
                    ElevatedButton(
                      onPressed: _ensureCameraPermission,
                      child: const Text('Allow Camera'),
                    ),
                  ],
                ),
              )
            : (_initError != null)
                ? Center(
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Text(_initError!),
                        const SizedBox(height: 12),
                        ElevatedButton(
                          onPressed: _ensureCameraPermission,
                          child: const Text('Retry'),
                        ),
                      ],
                    ),
                  )
                : (_controller == null || !_controller!.value.isInitialized || _initializing)
                    ? const Center(child: CircularProgressIndicator())
                    : Stack(
                        children: [
                          Positioned.fill(child: CameraPreview(_controller!)),
                          Positioned.fill(
                            child: Container(
                              decoration: const BoxDecoration(
                                gradient: LinearGradient(
                                  colors: [
                                    Color.fromARGB(70, 0, 0, 0),
                                    Color.fromARGB(10, 0, 0, 0),
                                  ],
                                  begin: Alignment.topCenter,
                                  end: Alignment.bottomCenter,
                                ),
                              ),
                            ),
                          ),
                          SafeArea(
                            child: Padding(
                              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                              child: Column(
                                children: [
                                  if (!_hasFace) ...[
                                    Container(
                                      width: double.infinity,
                                      padding: const EdgeInsets.all(12),
                                      decoration: BoxDecoration(
                                        color: const Color(0xAAFFF0F6),
                                        borderRadius: BorderRadius.circular(12),
                                      ),
                                      child: const Text(
                                        "Align your face and hold still. Good lighting helps us detect skin type and problems.",
                                        style: TextStyle(color: Color(0xFFB0005E)),
                                        textAlign: TextAlign.center,
                                      ),
                                    ),
                                    const SizedBox(height: 8),
                                  ],
                                  Wrap(
                                    alignment: WrapAlignment.center,
                                    spacing: 8,
                                    children: [
                                      ChoiceChip(
                                        label: const Text("Skin type "),
                                        selected: _scanMode == ScanMode.typeOnly,
                                        onSelected: (v) {
                                          if (!v) return;
                                          _applyMode(ScanMode.typeOnly);
                                        },
                                      ),
                                      ChoiceChip(
                                        label: const Text("Skin problem "),
                                        selected: _scanMode == ScanMode.problemOnly,
                                        onSelected: (v) {
                                          if (!v) return;
                                          _applyMode(ScanMode.problemOnly);
                                        },
                                      ),
                                    ],
                                  ),
                                  const SizedBox(height: 8),
                                  const Spacer(),
                                  Container(
                                    width: double.infinity,
                                    padding: const EdgeInsets.all(12),
                                    decoration: BoxDecoration(
                                      color: const Color(0xCCFFFFFF),
                                      borderRadius: BorderRadius.circular(14),
                                    ),
                                    child: Column(
                                      crossAxisAlignment: CrossAxisAlignment.start,
                                      children: [
                                        Text(
                                          _lastPrediction,
                                          style: const TextStyle(fontWeight: FontWeight.w600),
                                          textAlign: TextAlign.left,
                                        ),
                                        const SizedBox(height: 6),
                                        _ResultsPanel(
                                          skinType: _detectedSkinType,
                                          skinProblem: _detectedSkinProblem,
                                          pendingProblemScan: _pendingProblemScan || (_detectProblem && _scanMode != ScanMode.typeOnly),
                                          routineText: _routineText,
                                          scanMode: _scanMode,
                                        ),
                                        const SizedBox(height: 6),
                                        Text(
                                          "Camera: $_cameraInfo\nSeen: $_framesSeen | Attempts: $_inferenceAttempts | Predictions: $_framesProcessed\nDebug: $_debugInfo",
                                          style: const TextStyle(fontSize: 11, color: Colors.grey),
                                        ),
                                      ],
                                    ),
                                  ),
                                ],
                              ),
                            ),
                          ),
                        ],
                      ),
      ),
    );
  }
}

class _ResultsPanel extends StatelessWidget {
  final String? skinType;
  final String? skinProblem;
  final bool pendingProblemScan;
  final String routineText;
  final ScanMode scanMode;

  const _ResultsPanel({
    required this.skinType,
    required this.skinProblem,
    required this.pendingProblemScan,
    required this.routineText,
    required this.scanMode,
  });

  String _problemDisplay() {
    if (scanMode == ScanMode.typeOnly) return "Not requested";
    if (skinProblem != null && skinProblem!.isNotEmpty) return skinProblem!;
    if (pendingProblemScan) return "Scanning for skin problems...";
    return "Not detected yet";
  }

  String _typeDisplay() {
    if (skinType != null && skinType!.isNotEmpty) return skinType!;
    return "Scanning for skin type...";
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: const Color(0xFFFFEEF5),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            "Detected Skin Problems",
            style: TextStyle(
              fontSize: 15,
              fontWeight: FontWeight.w700,
              color: Color(0xFFB0005E),
            ),
          ),
          const SizedBox(height: 4),
          Text(_problemDisplay()),
          const SizedBox(height: 10),
          const Text(
            "Predicted Skin Type",
            style: TextStyle(
              fontSize: 15,
              fontWeight: FontWeight.w700,
              color: Color(0xFFB0005E),
            ),
          ),
          const SizedBox(height: 4),
          Text(_typeDisplay()),
          const SizedBox(height: 10),
          const Text(
            "Suggested Routine",
            style: TextStyle(
              fontSize: 15,
              fontWeight: FontWeight.w700,
              color: Color(0xFFB0005E),
            ),
          ),
          const SizedBox(height: 4),
          Text(
            routineText,
            style: const TextStyle(fontSize: 14),
          ),
          const SizedBox(height: 10),
          if ((skinType != null && scanMode == ScanMode.typeOnly) ||
              (skinProblem != null && scanMode == ScanMode.problemOnly) ||
              (skinType != null && skinProblem != null))
            SizedBox(
              width: double.infinity,
              child: ElevatedButton(
                onPressed: () {
                  // Done -> return to home
                  Navigator.popUntil(
                    context,
                    (route) => route.isFirst,
                  );
                },
                style: ElevatedButton.styleFrom(
                  backgroundColor: const Color(0xFFB0005E),
                  foregroundColor: Colors.white,
                ),
                child: const Text("Done"),
              ),
            ),
        ],
      ),
    );
  }
}
