import 'dart:async';
import 'dart:typed_data';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show DeviceOrientation;
import 'package:image/image.dart' as img;
import 'makeup_service.dart';
import 'package:google_mlkit_face_mesh/google_mlkit_face_mesh.dart';
import 'package:google_mlkit_commons/google_mlkit_commons.dart';

class MakeupCameraPage extends StatefulWidget {
  const MakeupCameraPage({super.key});

  @override
  State<MakeupCameraPage> createState() => _MakeupCameraPageState();
}

class _MakeupCameraPageState extends State<MakeupCameraPage> {
  CameraController? _controller;
  bool _cameraReady = false;
  Uint8List? _serverOutput;
  String? _error;
  Size? _lastPreviewSizeLogged;
  int _lastRotationLogged = -1;
  bool _sendingFrame = false;
  DateTime _lastSent = DateTime.fromMillisecondsSinceEpoch(0);

  @override
  void initState() {
    super.initState();
    _initCamera();
  }

  Future<void> _initCamera() async {
    try {
      final cameras = await availableCameras();
      if (cameras.isEmpty) {
        setState(() => _error = 'No camera found');
        return;
      }
      final preferred = cameras.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.front,
        orElse: () => cameras.first,
      );
      final controller = CameraController(
        preferred,
        ResolutionPreset.medium,
        enableAudio: false,
      );
      await controller.initialize();
      await controller.startImageStream(_processCameraImage);
      if (!mounted) return;
      setState(() {
        _controller = controller;
        _cameraReady = true;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = 'Camera error: $e');
    }
  }

  @override
  void dispose() {
    _controller?.dispose();
    super.dispose();
  }

  void _processCameraImage(CameraImage image) async {
    if (_controller == null || !_controller!.value.isInitialized) return;
    final now = DateTime.now();
    if (_sendingFrame || now.difference(_lastSent) < const Duration(milliseconds: 180)) return;
    _lastSent = now;
    _sendingFrame = true;

    try {
      // Rotate to upright portrait locally (90 deg). Mirror stays off to avoid left flip.
      const int rotationDeg = 90;
      const bool mirrorParam = false;
      _logDebug(rotationDeg, mirrorParam);

      // Rotate locally before send so server gets upright frame.
      final jpegBytes = _cameraImageToJpeg(image, rotationDeg, mirrorHoriz: false);

      final out = await sendMakeupRequest(
        jpegBytes: jpegBytes,
        style: 'simple',
        shadowOpacity: 0.55,
        blushOpacity: 0.28,
        lipOpacity: 0.60,
        highlightOpacity: 0.22,
        mirror: mirrorParam,
        rotation: 0, // already rotated locally
      ).timeout(const Duration(seconds: 6));

      if (!mounted) return;
      setState(() => _serverOutput = out);
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = e.toString());
    } finally {
      _sendingFrame = false;
    }
  }

  Widget _buildCoverPreview() {
    final c = _controller!;
    final size = c.value.previewSize;
    if (size != null && _lastPreviewSizeLogged != size) {
      _lastPreviewSizeLogged = size;
      debugPrint('[makeup] previewSize: ${size.width} x ${size.height}');
    }
    return Stack(
      fit: StackFit.expand,
      children: [
        CameraPreview(c),
        if (_serverOutput != null)
          IgnorePointer(
            child: Image.memory(
              _serverOutput!,
              fit: BoxFit.cover,
              gaplessPlayback: true,
            ),
          ),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    final previewReady = _cameraReady && _controller != null && _controller!.value.isInitialized;
    return Scaffold(
      appBar: AppBar(
        title: const Text('Virtual Makeup'),
        backgroundColor: Colors.white,
        foregroundColor: const Color(0xFFB0005E),
      ),
      body: previewReady
          ? Stack(
              fit: StackFit.expand,
              children: [
                _buildCoverPreview(),
                if (_error != null)
                  Positioned(
                    left: 16,
                    right: 16,
                    bottom: 70,
                    child: Text(
                      _error!,
                      style: const TextStyle(color: Colors.red),
                      textAlign: TextAlign.center,
                    ),
                  ),
              ],
            )
          : const Center(child: CircularProgressIndicator()),
    );
  }

  int _computeRotationDeg() {
    final cam = _controller!.description;
    final sensor = cam.sensorOrientation; // e.g., 90 on many Android devices
    final deviceOrientation = _controller!.value.deviceOrientation;
    final int deviceDeg = switch (deviceOrientation) {
      DeviceOrientation.portraitUp => 0,
      DeviceOrientation.landscapeLeft => 90,
      DeviceOrientation.portraitDown => 180,
      DeviceOrientation.landscapeRight => 270,
      _ => 0,
    };
    final bool isFront = cam.lensDirection == CameraLensDirection.front;
    // Clockwise rotation to make the image upright for ML/processing
    final int rotationDeg = isFront
        ? (sensor + deviceDeg) % 360
        : (sensor - deviceDeg + 360) % 360;
    return rotationDeg;
  }

  final _meshDetector = FaceMeshDetector(
  options: FaceMeshDetectorOptions(
    enableContours: true,
    enableClassification: false,
  ),
);

  void _logDebug(int rotationDeg, bool mirror) {
    if (_lastRotationLogged == rotationDeg) return;
    _lastRotationLogged = rotationDeg;
    final cam = _controller!;
    final size = cam.value.previewSize;
    debugPrint('[makeup] sensorOrientation=${cam.description.sensorOrientation} '
        'deviceOrientation=${cam.value.deviceOrientation} '
        'computedRotation=$rotationDeg mirror=$mirror '
        'previewSize=${size?.width}x${size?.height}');
  }

  Uint8List _cameraImageToJpeg(CameraImage image, int rotationDeg, {required bool mirrorHoriz}) {
    final int width = image.width;
    final int height = image.height;
    final Plane planeY = image.planes[0];
    final Plane planeU = image.planes[1];
    final Plane planeV = image.planes[2];

    final int strideY = planeY.bytesPerRow;
    final int strideU = planeU.bytesPerRow;
    final int strideV = planeV.bytesPerRow;

    final Uint8List bytesY = planeY.bytes;
    final Uint8List bytesU = planeU.bytes;
    final Uint8List bytesV = planeV.bytes;

    final img.Image rgb = img.Image(width, height);

    for (int y = 0; y < height; y++) {
      final int uvRowU = (y >> 1) * strideU;
      final int uvRowV = (y >> 1) * strideV;
      for (int x = 0; x < width; x++) {
        final int yIndex = y * strideY + x;
        final int uvIndex = (x >> 1);

        final int Y = bytesY[yIndex];
        final int U = bytesU[uvRowU + uvIndex];
        final int V = bytesV[uvRowV + uvIndex];

        final int c = Y - 16;
        final int d = U - 128;
        final int e = V - 128;

        int r = (298 * c + 409 * e + 128) >> 8;
        int g = (298 * c - 100 * d - 208 * e + 128) >> 8;
        int b = (298 * c + 516 * d + 128) >> 8;

        if (r < 0) r = 0; else if (r > 255) r = 255;
        if (g < 0) g = 0; else if (g > 255) g = 255;
        if (b < 0) b = 0; else if (b > 255) b = 255;

        rgb.setPixel(x, y, img.getColor(r, g, b));
      }
    }

    img.Image oriented = rgb;
    if (rotationDeg == 90) {
      oriented = img.copyRotate(rgb, 90);
    } else if (rotationDeg == 270) {
      oriented = img.copyRotate(rgb, -90);
    } else if (rotationDeg == 180) {
      oriented = img.copyRotate(rgb, 180);
    }
    if (mirrorHoriz) {
      oriented = img.flipHorizontal(oriented);
    }

    return Uint8List.fromList(img.encodeJpg(oriented, quality: 90));
  }
}
