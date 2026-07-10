"""
Baixa a pasta 00_Dados/ (coletas do scraper KaBuM) do Google Drive.

Uso:
    python baixar_dados.py

Requer o pacote `gdown` (já incluso em requirements.txt).
"""
from __future__ import annotations

import pathlib
import sys

DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1ZSOr9PP7XvwfqOHyXkj7JwfS_0cX-uJU?usp=sharing"
DESTINO = pathlib.Path(__file__).parent / "00_Dados"


def main() -> None:
    try:
        import gdown
    except ImportError:
        print("O pacote 'gdown' não está instalado.")
        print("Rode: pip install -r requirements.txt")
        sys.exit(1)

    if DESTINO.exists() and any(DESTINO.iterdir()):
        print(f"A pasta '{DESTINO}' já existe e não está vazia.")
        resposta = input("Baixar mesmo assim (pode sobrescrever arquivos)? [s/N] ").strip().lower()
        if resposta != "s":
            print("Cancelado.")
            return

    DESTINO.mkdir(parents=True, exist_ok=True)

    print(f"Baixando dados de:\n  {DRIVE_FOLDER_URL}\npara:\n  {DESTINO}\n")
    gdown.download_folder(
        url=DRIVE_FOLDER_URL,
        output=str(DESTINO),
        quiet=False,
        use_cookies=False,
    )

    subpastas = sorted(p for p in DESTINO.iterdir() if p.is_dir())
    total_arquivos = sum(1 for p in DESTINO.rglob("*") if p.is_file())

    print("\n✓ Download concluído.")
    print(f"  {len(subpastas)} pasta(s) de coleta encontrada(s):")
    for p in subpastas:
        n = sum(1 for _ in p.glob("*.csv"))
        print(f"    - {p.name}  ({n} arquivo(s) csv)")
    print(f"  {total_arquivos} arquivo(s) no total em '{DESTINO}'.")

    if not subpastas:
        print(
            "\nAVISO: nenhuma subpasta foi baixada. Confirme se a pasta do "
            "Drive está compartilhada como 'Qualquer pessoa com o link — "
            "Leitor', ou baixe manualmente pelo link acima."
        )


if __name__ == "__main__":
    main()
