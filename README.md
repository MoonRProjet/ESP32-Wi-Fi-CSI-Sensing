# ESP32 & Jetson Nano : Détection de mouvement par Wi-Fi (CSI)

## Description
Ce projet utilise un ESP32 pour capturer les perturbations des ondes Wi-Fi (CSI) dans une pièce. L'ESP32 envoie ces données brutes par câble USB à une Jetson Nano, qui filtre les informations et calcule un score de mouvement en temps réel.

## Architecture Matérielle
1. **Smartphone (Point d'accès)** : Émet le signal Wi-Fi.
2. **ESP32 (Récepteur)** : Sniffe les paquets Wi-Fi et extrait les données CSI.
3. **Jetson Nano (Traitement)** : Lit la liaison série, filtre par adresse MAC et génère les logs CSV.

## Structure du projet
- `esp32_receiver/` : Code C pour l'ESP32 (Framework ESP-IDF v5.5.4).
- `jetson_processor/` : Scripts Python pour la Jetson Nano.

## Guide d'installation

### Étape 1 : Flasher l'ESP32
Connecte l'ESP32 à ton PC et ouvre un terminal dans `esp32_receiver/` :
```bash
idf.py set-target esp32
idf.py build flash
