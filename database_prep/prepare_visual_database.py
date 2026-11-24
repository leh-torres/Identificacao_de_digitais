#!/usr/bin/env python3
"""
Script para preparar base de dados AFIS com imagens + minúcias
Copia imagens PNG junto com arquivos XYT para facilitar visualização
"""

import argparse
import shutil
from pathlib import Path
from tqdm import tqdm
import json

def prepare_visual_database(image_dir: str, xyt_dir: str, output_dir: str):
    """
    Prepara base unificada com imagens e minúcias
    
    Args:
        image_dir: Diretório com imagens PNG originais
        xyt_dir: Diretório com arquivos .xyt extraídos
        output_dir: Diretório de saída (conterá PNG + XYT)
    """
    image_path = Path(image_dir)
    xyt_path = Path(xyt_dir)
    output_path = Path(output_dir)
    
    if not image_path.exists():
        raise FileNotFoundError(f"Diretório de imagens não encontrado: {image_dir}")
    
    if not xyt_path.exists():
        raise FileNotFoundError(f"Diretório de minúcias não encontrado: {xyt_dir}")
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Lista arquivos
    png_files = {f.stem: f for f in image_path.glob("*.png")}
    xyt_files = {f.stem: f for f in xyt_path.glob("*.xyt")}
    
    print(f"📁 Imagens encontradas: {len(png_files)}")
    print(f"📁 Minúcias encontradas: {len(xyt_files)}")
    
    # Encontra pares correspondentes
    common_ids = set(png_files.keys()) & set(xyt_files.keys())
    
    if not common_ids:
        print("❌ Nenhum par correspondente encontrado!")
        print("\nExemplos de IDs em imagens:")
        for stem in list(png_files.keys())[:5]:
            print(f"  - {stem}")
        print("\nExemplos de IDs em minúcias:")
        for stem in list(xyt_files.keys())[:5]:
            print(f"  - {stem}")
        return
    
    print(f"✅ Pares correspondentes: {len(common_ids)}\n")
    
    # Copia arquivos
    copied = 0
    errors = []
    
    for file_id in tqdm(common_ids, desc="Copiando arquivos"):
        try:
            # Copia PNG
            src_png = png_files[file_id]
            dst_png = output_path / src_png.name
            shutil.copy2(src_png, dst_png)
            
            # Copia XYT
            src_xyt = xyt_files[file_id]
            dst_xyt = output_path / src_xyt.name
            shutil.copy2(src_xyt, dst_xyt)
            
            copied += 1
        except Exception as e:
            errors.append({'id': file_id, 'error': str(e)})
    
    # Relatório
    print(f"\n✅ Base visual preparada!")
    print(f"   Pares copiados: {copied}")
    print(f"   Diretório: {output_path}")
    
    if errors:
        print(f"\n⚠️  Erros: {len(errors)}")
        for err in errors[:5]:
            print(f"   - {err['id']}: {err['error']}")
    
    # Salva relatório
    report = {
        'total_images': len(png_files),
        'total_minutiae': len(xyt_files),
        'pairs_copied': copied,
        'errors': len(errors),
        'output_dir': str(output_path)
    }
    
    report_path = output_path / "database_info.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"   Relatório: {report_path}")
    
    # Verifica estrutura final
    final_png = len(list(output_path.glob("*.png")))
    final_xyt = len(list(output_path.glob("*.xyt")))
    
    print(f"\n📊 Estrutura final:")
    print(f"   PNG: {final_png}")
    print(f"   XYT: {final_xyt}")
    print(f"   Status: {'✅ OK' if final_png == final_xyt else '⚠️ DESBALANCEADO'}")

def verify_database(db_dir: str):
    """Verifica integridade da base visual"""
    db_path = Path(db_dir)
    
    if not db_path.exists():
        print(f"❌ Diretório não encontrado: {db_dir}")
        return
    
    png_files = {f.stem: f for f in db_path.glob("*.png")}
    xyt_files = {f.stem: f for f in db_path.glob("*.xyt")}
    
    print(f"📊 Verificação da Base Visual")
    print(f"   PNG: {len(png_files)}")
    print(f"   XYT: {len(xyt_files)}")
    
    # Verifica pares
    only_png = set(png_files.keys()) - set(xyt_files.keys())
    only_xyt = set(xyt_files.keys()) - set(png_files.keys())
    complete_pairs = set(png_files.keys()) & set(xyt_files.keys())
    
    print(f"   Pares completos: {len(complete_pairs)}")
    
    if only_png:
        print(f"\n⚠️  PNG sem XYT: {len(only_png)}")
        for stem in list(only_png)[:5]:
            print(f"      - {stem}")
    
    if only_xyt:
        print(f"\n⚠️  XYT sem PNG: {len(only_xyt)}")
        for stem in list(only_xyt)[:5]:
            print(f"      - {stem}")
    
    if complete_pairs:
        print(f"\n✅ Base válida para matching visual")

def main():
    parser = argparse.ArgumentParser(
        description='Preparar base de dados AFIS com visualização'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Comando')
    
    # Comando: prepare
    prepare_parser = subparsers.add_parser(
        'prepare',
        help='Prepara base visual (PNG + XYT)'
    )
    prepare_parser.add_argument(
        'image_dir',
        help='Diretório com imagens PNG originais'
    )
    prepare_parser.add_argument(
        'xyt_dir',
        help='Diretório com arquivos .xyt'
    )
    prepare_parser.add_argument(
        'output_dir',
        help='Diretório de saída (PNG + XYT juntos)'
    )
    
    # Comando: verify
    verify_parser = subparsers.add_parser(
        'verify',
        help='Verifica integridade da base visual'
    )
    verify_parser.add_argument(
        'db_dir',
        help='Diretório da base visual'
    )
    
    args = parser.parse_args()
    
    if args.command == 'prepare':
        prepare_visual_database(args.image_dir, args.xyt_dir, args.output_dir)
    elif args.command == 'verify':
        verify_database(args.db_dir)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()


# Exemplos de uso:
"""
# 1. Preparar base visual
python prepare_visual_database.py prepare \
    /path/to/original/images \
    /path/to/extracted/xyt \
    /path/to/visual/database

# 2. Verificar base
python prepare_visual_database.py verify /path/to/visual/database
"""