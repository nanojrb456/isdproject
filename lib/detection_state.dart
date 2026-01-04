class DetectionState {
  static bool hasRoutine = false;
  static String routineText = "Routine will appear here after detection.";

  static void updateRoutine(String text) {
    routineText = text;
    hasRoutine = text.trim().isNotEmpty;
  }
}
