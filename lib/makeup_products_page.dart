import 'package:flutter/material.dart';

class MakeupProductsPage extends StatelessWidget {
  const MakeupProductsPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Makeup Products'),
        backgroundColor: Colors.white,
        foregroundColor: const Color(0xFFB0005E),
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: const [
          _ProductCard(
            title: 'Hydrating Primer',
            description: 'Blurs pores and grips makeup with a soft glow.',
            tag: 'Prep',
          ),
          _ProductCard(
            title: 'Lightweight Skin Tint',
            description: 'Sheer coverage with dewy finish.',
            tag: 'Base',
          ),
          _ProductCard(
            title: 'Cream Blush',
            description: 'Blendable flush that melts into skin.',
            tag: 'Color',
          ),
          _ProductCard(
            title: 'Tubing Mascara',
            description: 'Smudge-proof length and definition.',
            tag: 'Eyes',
          ),
          _ProductCard(
            title: 'Nourishing Lip Oil',
            description: 'Tinted shine with conditioning oils.',
            tag: 'Lips',
          ),
        ],
      ),
    );
  }
}

class _ProductCard extends StatelessWidget {
  final String title;
  final String description;
  final String tag;

  const _ProductCard({
    required this.title,
    required this.description,
    required this.tag,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(14),
        boxShadow: [
          BoxShadow(
            color: Colors.pink.withOpacity(0.06),
            blurRadius: 10,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Row(
        children: [
          Container(
            padding: const EdgeInsets.all(10),
            decoration: BoxDecoration(
              color: const Color(0xFFFFEEF5),
              borderRadius: BorderRadius.circular(12),
            ),
            child: Text(
              tag,
              style: const TextStyle(
                fontWeight: FontWeight.bold,
                color: Color(0xFFB0005E),
              ),
            ),
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
                  style: TextStyle(color: Colors.grey.shade800),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
