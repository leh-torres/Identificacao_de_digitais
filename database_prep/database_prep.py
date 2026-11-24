#!/usr/bin/env python3
"""
Script para preparar base de dados AFIS
- Extrai minúcias de todas as imagens em um diretório
- Cria estrutura otimizada para buscas rápidas
"""

import argparse
from pathlib import Path
from tqdm import tqdm
import json
from afis_engine import AfisEngine
import multiprocessing as mp
from functools import partial

def process_image(image_path: Path, afis: AfisEngine, output_dir: Path) -> dict:
    """Processa uma imagem e salva as minúcias"""
    try:
        # Extrai minúcias
        xyt_path = afis.extract_minutiae(str(image_path))
        
        # Move para diretório de saída com nome original
        output_file = output_dir / f"{image_path.stem}.xyt"
        Path(xyt_path).rename(output_file)
        
        return {
            'status': 'success',
            'image': image_path.name,
            'xyt': str(output_file)
        }
    except Exception as e:
        return {
            'status': 'error',
            'image': image_path.name,
            'error': str(e)
        }

def prepare_database(input_dir: str, output_dir: str, num_cores: int = 4):
    """
    Processa um diretório de imagens e cria base de minúcias
    
    Args:
        input_dir: Diretório com imagens PNG
        output_dir: Diretório de saída para arquivos .xyt
        num_cores: Número de núcleos para processamento paralelo
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Diretório não encontrado: {input_dir}")
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Lista todas as imagens
    image_files = list(input_path.glob("*.png"))
    
    if not image_files:
        print("❌ Nenhuma imagem PNG encontrada")
        return
    
    print(f"📁 Encontradas {len(image_files)} imagens")
    print(f"⚙️  Usando {num_cores} núcleos\n")
    
    # Cria instância do AFIS
    afis = AfisEngine(num_cores=num_cores)
    
    # Processa em paralelo
    results = []
    
    with mp.Pool(processes=num_cores) as pool:
        process_func = partial(process_image, afis=afis, output_dir=output_path)
        
        for result in tqdm(
            pool.imap_unordered(process_func, image_files),
            total=len(image_files),
            desc="Extraindo minúcias"
        ):
            results.append(result)
    
    # Estatísticas
    success = sum(1 for r in results if r['status'] == 'success')
    errors = sum(1 for r in results if r['status'] == 'error')
    
    print(f"\n✅ Concluído!")
    print(f"   Sucesso: {success}")
    print(f"   Erros: {errors}")
    
    # Salva relatório
    report_path = output_path / "extraction_report.json"
    with open(report_path, 'w') as f:
        json.dump({
            'total': len(image_files),
            'success': success,
            'errors': errors,
            'results': results
        }, f, indent=2)
    
    print(f"   Relatório: {report_path}")
    
    # Lista erros se houver
    if errors > 0:
        print("\n⚠️  Erros encontrados:")
        for r in results:
            if r['status'] == 'error':
                print(f"   - {r['image']}: {r['error']}")

def verify_database(db_dir: str):
    """Verifica integridade da base de dados"""
    db_path = Path(db_dir)
    
    if not db_path.exists():
        print(f"❌ Diretório não encontrado: {db_dir}")
        return
    
    xyt_files = list(db_path.glob("*.xyt"))
    
    print(f"📊 Estatísticas da Base")
    print(f"   Total de registros: {len(xyt_files)}")
    
    # Analisa tamanhos
    sizes = [f.stat().st_size for f in xyt_files]
    if sizes:
        print(f"   Tamanho médio: {sum(sizes)/len(sizes):.0f} bytes")
        print(f"   Tamanho mín: {min(sizes)} bytes")
        print(f"   Tamanho máx: {max(sizes)} bytes")
    
    # Verifica arquivos válidos
    valid = 0
    invalid = []
    
    for f in xyt_files:
        try:
            with open(f, 'r') as file:
                lines = file.readlines()
                if len(lines) > 0:
                    valid += 1
                else:
                    invalid.append(f.name)
        except:
            invalid.append(f.name)
    
    print(f"   Arquivos válidos: {valid}")
    
    if invalid:
        print(f"   ⚠️  Arquivos inválidos: {len(invalid)}")
        for name in invalid[:5]:
            print(f"      - {name}")

def test_search(db_dir: str, probe_image: str, cores: int = 4):
    """Testa busca na base"""
    print(f"🔍 Testando busca na base")
    print(f"   Base: {db_dir}")
    print(f"   Probe: {probe_image}")
    print(f"   Núcleos: {cores}\n")
    
    afis = AfisEngine(num_cores=cores)
    
    # Extrai minúcias da probe
    print("Extraindo minúcias da probe...")
    probe_xyt = afis.extract_minutiae(probe_image)
    
    # Executa busca
    print("Executando busca...")
    matches = afis.one_many_match_cpu(
        probe_path=probe_xyt,
        db_path=db_dir,
        top_k=5,
        threshold=40
    )
    
    # Exibe resultados
    print(f"\n✅ Top {len(matches)} matches:")
    for i, match in enumerate(matches, 1):
        print(f"   #{i} - {match.candidate_id}")
        print(f"        Score: {match.score}")
        print(f"        Tempo: {match.time_ms:.2f}ms\n")

def main():
    parser = argparse.ArgumentParser(
        description='Ferramentas para gerenciar base AFIS'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Comando')
    
    # Comando: prepare
    prepare_parser = subparsers.add_parser(
        'prepare',
        help='Prepara base de dados extraindo minúcias'
    )
    prepare_parser.add_argument(
        'input_dir',
        help='Diretório com imagens PNG'
    )
    prepare_parser.add_argument(
        'output_dir',
        help='Diretório de saída para arquivos .xyt'
    )
    prepare_parser.add_argument(
        '--cores',
        type=int,
        default=4,
        help='Número de núcleos (padrão: 4)'
    )
    
    # Comando: verify
    verify_parser = subparsers.add_parser(
        'verify',
        help='Verifica integridade da base'
    )
    verify_parser.add_argument(
        'db_dir',
        help='Diretório da base de dados'
    )
    
    # Comando: test
    test_parser = subparsers.add_parser(
        'test',
        help='Testa busca na base'
    )
    test_parser.add_argument(
        'db_dir',
        help='Diretório da base de dados'
    )
    test_parser.add_argument(
        'probe_image',
        help='Imagem probe (PNG)'
    )
    test_parser.add_argument(
        '--cores',
        type=int,
        default=4,
        help='Número de núcleos (padrão: 4)'
    )
    
    args = parser.parse_args()
    
    if args.command == 'prepare':
        prepare_database(args.input_dir, args.output_dir, args.cores)
    elif args.command == 'verify':
        verify_database(args.db_dir)
    elif args.command == 'test':
        test_search(args.db_dir, args.probe_image, args.cores)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()


# Exemplos de uso:
"""
# 1. Preparar base de dados
python database_prep.py prepare /caminho/imagens /caminho/base --cores 8

# 2. Verificar base
python database_prep.py verify /caminho/base

# 3. Testar busca
python database_prep.py test /caminho/base probe.png --cores 4
"""