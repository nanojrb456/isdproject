import 'package:flutter/material.dart';
import 'makeup_camera_page.dart';

class MakeUpPage extends StatelessWidget {
  const MakeUpPage({super.key});

  void _open(BuildContext context, Widget page) {
    Navigator.push(context, MaterialPageRoute(builder: (_) => page));
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Makeup Looks'),
        backgroundColor: Colors.white,
        foregroundColor: const Color(0xFFB0005E),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Pick your style',
              style: TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.bold,
                color: Color(0xFFB0005E),
              ),
            ),
            const SizedBox(height: 12),
            _makeupCard(
              context,
              icon: Icons.star_rate_rounded,
              title: 'Glam Makeup',
              subtitle: 'Full glam with bold lips and contour.',
              onTap: () => _open(context, const MakeupCameraPage()),
            ),
            _makeupCard(
              context,
              icon: Icons.wb_sunny_outlined,
              title: 'Simple Makeup',
              subtitle: 'Light, breathable, and fresh.',
              onTap: () => _open(context, const MakeupCameraPage()),
            ),
            _makeupCard(
              context,
              icon: Icons.auto_awesome,
              title: 'Auto Makeup',
              subtitle: 'Powered by Elite model predictions.',
              onTap: () => _open(context, const MakeupCameraPage()),
            ),
          ],
        ),
      ),
    );
  }

  Widget _makeupCard(
    BuildContext context, {
    required IconData icon,
    required String title,
    required String subtitle,
    required VoidCallback onTap,
  }) {
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        boxShadow: [
          BoxShadow(
            color: Colors.pink.withOpacity(0.06),
            blurRadius: 12,
            offset: const Offset(0, 6),
          ),
        ],
      ),
      child: ListTile(
        leading: CircleAvatar(
          backgroundColor: const Color(0xFFFFEEF5),
          child: Icon(icon, color: const Color(0xFFB0005E)),
        ),
        title: Text(
          title,
          style: const TextStyle(fontWeight: FontWeight.w700),
        ),
        subtitle: Text(subtitle),
        trailing: const Icon(Icons.chevron_right),
        onTap: onTap,
      ),
    );
  }
}
