import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';
import 'package:http/http.dart' as http;


/// Sends a frame + makeup options to the local FastAPI server and returns the processed PNG bytes.
/// - rotation: clockwise sensor rotation (0/90/180/270)
/// - mirror: only set to true if the preview is not already mirrored.

Future<Uint8List> sendMakeupRequest({
  required Uint8List jpegBytes,
  required String style,
  required double shadowOpacity,
  required double blushOpacity,
  required double lipOpacity,
  required double highlightOpacity,
  required bool mirror,
  required int rotation, // 0, 90, 180, 270
  String? shadowColor,
  String? blushColor,
  String? lipColor,
  String? highlightColor,
  String serverBase = 'http://10.0.2.2:8000',
  Duration timeout = const Duration(seconds: 4),
  http.Client? client,
}) async {
  if (rotation != 0 && rotation != 90 && rotation != 180 && rotation != 270) {
    throw ArgumentError('rotation must be 0, 90, 180 or 270 degrees.');
  }

  final b64 = base64Encode(jpegBytes);
  final body = {
    "image_base64": b64,
    "style": style,
    "shadow_opacity": shadowOpacity,
    "blush_opacity": blushOpacity,
    "lip_opacity": lipOpacity,
    "highlight_opacity": highlightOpacity,
    "mirror": mirror,
    "rotate": rotation,
    "debug": true,
    if (shadowColor != null) "shadow_color": shadowColor,
    if (blushColor != null) "blush_color": blushColor,
    if (lipColor != null) "lip_color": lipColor,
    if (highlightColor != null) "highlight_color": highlightColor,
  };

  final http.Client httpClient = client ?? http.Client();
  try {
    final resp = await httpClient
        .post(
          Uri.parse('$serverBase/makeup'),
          headers: {'Content-Type': 'application/json'},
          body: jsonEncode(body),
        )
        .timeout(timeout);
    if (resp.statusCode != 200) {
      throw Exception('Makeup server error: ${resp.statusCode} ${resp.body}');
    }
    final data = jsonDecode(resp.body);
    return base64Decode(data['image_base64']);
  } on TimeoutException {
    throw Exception('Makeup server timed out after ${timeout.inMilliseconds} ms');
  } finally {
    if (client == null) {
      httpClient.close();
    }
  }
}

/// Lightweight throttler to avoid flooding the server (drops frames when busy).
class MakeupApiThrottler {
  MakeupApiThrottler({
    this.minInterval = const Duration(milliseconds: 160),
    this.serverBase = 'http://10.0.2.2:8000',
  });

  final Duration minInterval;
  final String serverBase;
  DateTime _lastSent = DateTime.fromMillisecondsSinceEpoch(0);
  bool _inFlight = false;

  Future<Uint8List?> sendIfReady({
    required Uint8List jpegBytes,
    required String style,
    required double shadowOpacity,
    required double blushOpacity,
    required double lipOpacity,
    required double highlightOpacity,
    required bool mirror,
    required int rotation,
  }) async {
    final now = DateTime.now();
    if (_inFlight || now.difference(_lastSent) < minInterval) return null;
    _lastSent = now;
    _inFlight = true;
    try {
      return await sendMakeupRequest(
        jpegBytes: jpegBytes,
        style: style,
        shadowOpacity: shadowOpacity,
        blushOpacity: blushOpacity,
        lipOpacity: lipOpacity,
        highlightOpacity: highlightOpacity,
        mirror: mirror,
        rotation: rotation,
        serverBase: serverBase,
      );
    } finally {
      _inFlight = false;
    }
  }
}
