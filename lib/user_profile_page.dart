import 'package:flutter/material.dart';

class UserProfilePage extends StatelessWidget {
  const UserProfilePage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Your Profile'),
        backgroundColor: Colors.white,
        foregroundColor: const Color(0xFFB0005E),
      ),
      body: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                const CircleAvatar(
                  radius: 32,
                  backgroundColor: Color(0xFFFF9FCF),
                  child: Icon(
                    Icons.person,
                    color: Colors.white,
                    size: 32,
                  ),
                ),
                const SizedBox(width: 12),
                Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: const [
                    Text(
                      'Noura',
                      style: TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                    Text(
                      'Skin care + makeup enthusiast',
                      style: TextStyle(color: Colors.grey),
                    ),
                  ],
                ),
              ],
            ),
            const SizedBox(height: 24),
            const Text(
              'Daily progress',
              style: TextStyle(
                fontSize: 16,
                fontWeight: FontWeight.bold,
                color: Color(0xFFB0005E),
              ),
            ),
            const SizedBox(height: 10),
            _ProgressRow(
              title: 'Skin care routine',
              value: 0.6,
              color: Colors.pink,
              subtitle: '3 of 5 steps done today',
            ),
            const SizedBox(height: 12),
            _ProgressRow(
              title: 'Makeup routine',
              value: 0.3,
              color: Colors.purple,
              subtitle: 'Foundation and blush applied',
            ),
            const SizedBox(height: 24),
            const Text(
              'Past tasks',
              style: TextStyle(
                fontSize: 16,
                fontWeight: FontWeight.bold,
                color: Color(0xFFB0005E),
              ),
            ),
            const SizedBox(height: 10),
            Expanded(
              child: ListView(
                children: const [
                  _TaskItem(
                    title: 'Completed morning routine',
                    date: 'Today, 8:10 AM',
                    done: true,
                  ),
                  _TaskItem(
                    title: 'Logged evening moisturizer',
                    date: 'Yesterday, 9:15 PM',
                    done: true,
                  ),
                  _TaskItem(
                    title: 'Tried glam makeup (preview)',
                    date: 'Yesterday, 7:00 PM',
                    done: false,
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _ProgressRow extends StatelessWidget {
  final String title;
  final double value;
  final Color color;
  final String subtitle;

  const _ProgressRow({
    required this.title,
    required this.value,
    required this.color,
    required this.subtitle,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(title, style: const TextStyle(fontWeight: FontWeight.w600)),
        const SizedBox(height: 6),
        LinearProgressIndicator(
          value: value,
          color: color,
          backgroundColor: Colors.grey.shade200,
          minHeight: 8,
        ),
        const SizedBox(height: 4),
        Text(subtitle, style: TextStyle(color: Colors.grey.shade700)),
      ],
    );
  }
}

class _TaskItem extends StatelessWidget {
  final String title;
  final String date;
  final bool done;

  const _TaskItem({
    required this.title,
    required this.date,
    required this.done,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(bottom: 10),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(12),
        boxShadow: [
          BoxShadow(
            color: Colors.pink.withOpacity(0.05),
            blurRadius: 8,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Row(
        children: [
          Icon(
            done ? Icons.check_circle : Icons.radio_button_unchecked,
            color: done ? const Color(0xFFB0005E) : Colors.grey,
          ),
          const SizedBox(width: 10),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: const TextStyle(fontWeight: FontWeight.w600),
                ),
                const SizedBox(height: 4),
                Text(date, style: TextStyle(color: Colors.grey.shade700)),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
