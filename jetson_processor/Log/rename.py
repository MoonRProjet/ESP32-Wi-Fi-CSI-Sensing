#!/usr/bin/env python3

import os
from pathlib import Path

# ===== À MODIFIER =====
FOLDER = "./."   # dossier contenant les fichiers à renommer
PREFIX = "1_moving_"        # préfixe du nouveau nom
START = 1                      # numéro de départ
PADDING = 2                    # 01, 02, 03...
EXTENSION = ".csv"             # extension à cibler
# ======================

def main():
    folder = Path(FOLDER)

    if not folder.exists() or not folder.is_dir():
        print(f"Dossier introuvable : {folder}")
        return

    files = sorted([f for f in folder.iterdir() if f.is_file() and f.suffix.lower() == EXTENSION])

    if not files:
        print(f"Aucun fichier {EXTENSION} trouvé dans {folder}")
        return

    print("Fichiers trouvés :")
    for f in files:
        print(" -", f.name)

    # Étape 1 : renommage temporaire pour éviter les collisions
    temp_files = []
    for i, f in enumerate(files):
        temp_name = folder / f"__tmp_rename_{i}{f.suffix}"
        f.rename(temp_name)
        temp_files.append(temp_name)

    # Étape 2 : renommage final
    for idx, f in enumerate(temp_files, start=START):
        new_name = f"{PREFIX}{str(idx).zfill(PADDING)}{EXTENSION}"
        new_path = folder / new_name
        f.rename(new_path)
        print(f"Renommé : {f.name} -> {new_name}")

    print("Renommage terminé.")

if __name__ == "__main__":
    main()
