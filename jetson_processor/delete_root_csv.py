#!/usr/bin/env python3
from pathlib import Path

ROOT = Path.home() / "Melik_Project"

csv_files = sorted([p for p in ROOT.glob("*.csv") if p.is_file()])

if not csv_files:
    print("Aucun fichier CSV trouvé à la racine de Melik_Project.")
    raise SystemExit(0)

print("Fichiers CSV qui vont être supprimés :")
for f in csv_files:
    print(f" - {f.name}")

confirm = input("\nSupprimer ces fichiers ? (yes/no) : ").strip().lower()
if confirm != "yes":
    print("Annulé.")
    raise SystemExit(0)

deleted = 0
for f in csv_files:
    f.unlink()
    deleted += 1

print(f"\nTerminé : {deleted} fichier(s) supprimé(s).")
