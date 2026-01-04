import 'package:flutter/material.dart';

class SkinProductsPage extends StatelessWidget {
  const SkinProductsPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Skin Care Products'),
        backgroundColor: Colors.white,
        foregroundColor: const Color(0xFFB0005E),
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: const [
          _ProductCard(
            title: 'Gentle Gel Cleanser',
            description: 'pH-balanced, sulfate-free daily cleanser.',
            tag: 'AM/PM',
          ),
          _ProductCard(
            title: 'Niacinamide Serum 10%',
            description: 'Balances oil, calms redness, refines pores.',
            tag: 'AM/PM',
          ),
          _ProductCard(
            title: 'SPF 50 Mineral Sunscreen',
            description: 'Non-greasy protection with zinc oxide.',
            tag: 'AM',
          ),
          _ProductCard(
            title: 'Ceramide Moisturizer',
            description: 'Barrier-support cream for nightly repair.',
            tag: 'PM',
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
