import 'package:flutter/material.dart';


class SimpleMakeUpPage extends StatelessWidget {
  const SimpleMakeUpPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Simple MakeUp Filter'),
      ),
      body: Center(
        child: Text('Simple MakeUp Camera Filter'),
        // يمكنك إضافة الكود الخاص بالكاميرا والفلاتر الخاصة بـ Simple هنا
      ),
    );
  }
}