import 'package:flutter/material.dart';
import 'HomePage.dart';
import 'LoginPage.dart';
import 'MakeUpPage.dart';
import 'SignInPage.dart';
import 'SkinCarepage.dart';
import 'WelcomePage.dart';
import 'profile_about_page.dart';
import 'splash_screen.dart';
import 'user_profile_page.dart';
import 'virtual_makeup_page.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Makeuo',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFFFF6FAF)),
        useMaterial3: true,
        scaffoldBackgroundColor: const Color(0xFFFFF7FB),
      ),
      home: const SplashScreen(),
      routes: {
        '/welcome': (context) => const WelcomePage(),
        '/login': (context) => LoginPage(),
        '/signIn': (context) => SignInPage(),
        '/home': (context) => const HomePage(),
        '/skinCare': (context) => const SkinCarePage(),
        '/makeup': (context) => const MakeUpPage(),
        '/about': (context) => const ProfileAboutPage(),
        '/profile': (context) => const UserProfilePage(),
        '/virtualMakeup': (context) => const VirtualMakeUpPage(),
      },
    );
  }
}
