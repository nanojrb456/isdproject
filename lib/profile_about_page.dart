import 'package:flutter/material.dart';

class ProfileAboutPage extends StatelessWidget {
  const ProfileAboutPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('About the Project'),
        backgroundColor: Colors.white,
        foregroundColor: const Color(0xFFB0005E),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Center(
              child: Container(
                width: 110,
                height: 110,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  gradient: const LinearGradient(
                    colors: [Color(0xFFFF6FAF), Color(0xFFFF9FCF)],
                    begin: Alignment.topLeft,
                    end: Alignment.bottomRight,
                  ),
                  boxShadow: [
                    BoxShadow(
                      color: Colors.pink.withOpacity(0.2),
                      blurRadius: 18,
                      offset: const Offset(0, 10),
                    ),
                  ],
                ),
                child: const Icon(
                  Icons.favorite_outline,
                  color: Colors.white,
                  size: 52,
                ),
              ),
            ),
            const SizedBox(height: 20),
            const Text(
              'About Us',
              style: TextStyle(
                fontSize: 22,
                fontWeight: FontWeight.bold,
                color: Color(0xFFB0005E),
              ),
            ),
            const SizedBox(height: 8),
            Text(
              'Makeuo delivers personalized skin care and makeup routines. '
              'We scan your skin, detect type and problems, then suggest the best steps and products. '
              'For makeup lovers, we offer glam, simple, and auto looks powered by smart predictions.',
              style: TextStyle(color: Colors.grey.shade800, height: 1.5),
            ),
            const SizedBox(height: 20),
            const Text(
              'Features',
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.w700,
                color: Color(0xFFB0005E),
              ),
            ),
            const SizedBox(height: 10),
            _FeatureRow(
              icon: Icons.spa_outlined,
              title: 'Skin analysis',
              description: 'Camera-based scan to learn your skin type and concerns.',
            ),
            _FeatureRow(
              icon: Icons.checklist_rtl_outlined,
              title: 'Tailored routines',
              description: 'AM/PM steps with product ideas that fit your skin.',
            ),
            _FeatureRow(
              icon: Icons.brush_outlined,
              title: 'Virtual makeup',
              description: 'Pick glam, simple, or auto looks and preview with AR.',
            ),
            _FeatureRow(
              icon: Icons.insights_outlined,
              title: 'Progress tracking',
              description: 'Track daily completions and see your streaks.',
            ),
          ],
        ),
      ),
    );
  }
}

class _FeatureRow extends StatelessWidget {
  final IconData icon;
  final String title;
  final String description;

  const _FeatureRow({
    required this.icon,
    required this.title,
    required this.description,
  });

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8.0),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            padding: const EdgeInsets.all(10),
            decoration: BoxDecoration(
              color: const Color(0xFFFFEEF5),
              borderRadius: BorderRadius.circular(12),
            ),
            child: Icon(icon, color: const Color(0xFFB0005E)),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: const TextStyle(
                    fontWeight: FontWeight.w700,
                    fontSize: 15,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  description,
                  style: TextStyle(color: Colors.grey.shade800, height: 1.4),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
