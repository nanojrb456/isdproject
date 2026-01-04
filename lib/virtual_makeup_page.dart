import 'package:flutter/material.dart';

class VirtualMakeUpPage extends StatefulWidget {
  const VirtualMakeUpPage({super.key});

  @override
  State<VirtualMakeUpPage> createState() => _VirtualMakeUpPageState();
}

class _VirtualMakeUpPageState extends State<VirtualMakeUpPage> {
  String _style = 'simple';
  double _eyeshadowOpacity = 0.42;
  double _blushOpacity = 0.24;
  double _lipOpacity = 0.58;
  double _highlightOpacity = 0.24;
  bool _mirror = true; // front camera default
  int _rotation = 0;   // 0/90/180/270 based on device orientation

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Virtual Makeup (Preview)'),
        backgroundColor: Colors.white,
        foregroundColor: const Color(0xFFB0005E),
      ),
      body: Column(
        children: [
          Expanded(
            child: Container(
              margin: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(18),
                gradient: const LinearGradient(
                  colors: [Color(0xFFFFEEF5), Color(0xFFFFFFFF)],
                  begin: Alignment.topCenter,
                  end: Alignment.bottomCenter,
                ),
                border: Border.all(color: const Color(0xFFFF8CC8)),
              ),
              child: Center(
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: const [
                    Icon(
                      Icons.camera_alt_outlined,
                      size: 64,
                      color: Color(0xFFB0005E),
                    ),
                    SizedBox(height: 12),
                    Text(
                      'Live camera preview',
                      style: TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    SizedBox(height: 6),
                    Padding(
                      padding: EdgeInsets.symmetric(horizontal: 24.0),
                      child: Text(
                        'Glam, simple, and custom opacities will show here with AR makeup.',
                        textAlign: TextAlign.center,
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
          Padding(
            padding: const EdgeInsets.all(16.0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  'Style',
                  style: TextStyle(
                    fontWeight: FontWeight.bold,
                    fontSize: 16,
                    color: Color(0xFFB0005E),
                  ),
                ),
                const SizedBox(height: 8),
                Wrap(
                  spacing: 10,
                  children: [
                    ChoiceChip(
                      label: const Text('Simple'),
                      selected: _style == 'simple',
                      onSelected: (_) => setState(() => _style = 'simple'),
                    ),
                    ChoiceChip(
                      label: const Text('Glam'),
                      selected: _style == 'glam',
                      onSelected: (_) => setState(() => _style = 'glam'),
                    ),
                    ChoiceChip(
                      label: const Text('Custom'),
                      selected: _style == 'custom',
                      onSelected: (_) => setState(() => _style = 'custom'),
                    ),
                  ],
                ),
                const SizedBox(height: 16),
                _SliderRow(
                  label: 'Eyeshadow opacity',
                  value: _eyeshadowOpacity,
                  onChanged: (v) => setState(() => _eyeshadowOpacity = v),
                ),
                _SliderRow(
                  label: 'Blush opacity',
                  value: _blushOpacity,
                  onChanged: (v) => setState(() => _blushOpacity = v),
                ),
                _SliderRow(
                  label: 'Lipstick opacity',
                  value: _lipOpacity,
                  onChanged: (v) => setState(() => _lipOpacity = v),
                ),
                _SliderRow(
                  label: 'Highlighter opacity',
                  value: _highlightOpacity,
                  onChanged: (v) => setState(() => _highlightOpacity = v),
                ),
                const SizedBox(height: 12),
                SwitchListTile(
                  title: const Text('Mirror (front camera)'),
                  value: _mirror,
                  activeColor: const Color(0xFFB0005E),
                  onChanged: (v) => setState(() => _mirror = v),
                  subtitle: const Text('Keep makeup aligned with mirrored previews'),
                ),
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    const Text('Rotation'),
                    DropdownButton<int>(
                      value: _rotation,
                      items: const [
                        DropdownMenuItem(value: 0, child: Text('0°')),
                        DropdownMenuItem(value: 90, child: Text('90°')),
                        DropdownMenuItem(value: 180, child: Text('180°')),
                        DropdownMenuItem(value: 270, child: Text('270°')),
                      ],
                      onChanged: (v) => setState(() => _rotation = v ?? 0),
                    ),
                  ],
                ),
                const Text(
                  'What you will see soon',
                  style: TextStyle(
                    fontWeight: FontWeight.bold,
                    fontSize: 16,
                    color: Color(0xFFB0005E),
                  ),
                ),
                const SizedBox(height: 6),
                const Text('- Auto face detection and landmark masks'),
                const Text('- Style presets: Glam, Simple, Custom'),
                const Text('- Live opacity control and undo'),
                const Text('- Orientation fixes: rotate + mirror'),
                const SizedBox(height: 10),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _SliderRow extends StatelessWidget {
  const _SliderRow({
    required this.label,
    required this.value,
    required this.onChanged,
  });

  final String label;
  final double value;
  final ValueChanged<double> onChanged;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(
              label,
              style: const TextStyle(fontWeight: FontWeight.w600),
            ),
            Text('${(value * 100).round()}%'),
          ],
        ),
        Slider(
          min: 0,
          max: 1,
          value: value.clamp(0, 1),
          onChanged: onChanged,
          activeColor: const Color(0xFFB0005E),
          inactiveColor: const Color(0xFFFFC8E6),
        ),
        const SizedBox(height: 8),
      ],
    );
  }
}
