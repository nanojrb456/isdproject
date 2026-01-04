import 'package:flutter/material.dart';

class GlamMakeUpPage extends StatelessWidget {
  const GlamMakeUpPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Glam MakeUp Filter'),
      ),
      body: Center(
        child: Text('Glam MakeUp Camera Filter'),
        // يمكنك إضافة الكود الخاص بالكاميرا والفلاتر الخاصة بـ Glam هنا
      ),
    );
  }
}