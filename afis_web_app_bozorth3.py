"""
Integração do Bozorth3 GPU (artigo) com o sistema web AFIS
Adiciona novo modo: "bozorth3_gpu" na interface
"""

from flask import Flask, request, jsonify, render_template
from afis_engine import AfisEngine
from gpu_afis_engine import GPUAfisEngine
from bozorth3_gpu_engine import Bozorth3GPUEngine, CUDA_AVAILABLE as BOZORTH3_CUDA
from pathlib import Path
import base64
import time
import os

app = Flask(__name__)
app.template_folder = 'templates'
DATABASE_PATH = os.getenv("DATABASE_PATH", "/data/db_path")
# ============================================================================
# Configuração dos Engines
# ============================================================================

# Engine CPU tradicional
afis_cpu = AfisEngine(
    mindtct_path='/app/lib/nbis_build/bin/mindtct',
    bozorth3_path='/app/lib/nbis_build/bin/bozorth3',
    temp_dir='/app/lib/temp',
    num_cores=4
)

# Engine GPU (nossa implementação simplificada)
try:
    afis_gpu = GPUAfisEngine(
        bozorth3_path='/app/lib/nbis_build/bin/bozorth3',
        use_hybrid=True,
        gpu_batch_size=1000
    )
    gpu_available = afis_gpu.gpu_available
except Exception as e:
    print(f"⚠️  GPU (simplificada) não disponível: {e}")
    afis_gpu = None
    gpu_available = False

# Engine Bozorth3 GPU (implementação do artigo)
try:
    bozorth3_gpu = Bozorth3GPUEngine(
        distance_threshold=15.0,
        angle_threshold=20.0,
        threads_per_block=256
    )
    bozorth3_available = BOZORTH3_CUDA
except Exception as e:
    print(f"⚠️  Bozorth3 GPU não disponível: {e}")
    bozorth3_gpu = None
    bozorth3_available = False

print(f"\n📊 Status dos Engines:")
print(f"   CPU: ✅")
print(f"   GPU (simplificada): {'✅' if gpu_available else '❌'}")
print(f"   Bozorth3 GPU (artigo): {'✅' if bozorth3_available else '❌'}")

# ============================================================================
# Routes
# ============================================================================

@app.route("/")
def home():
    return render_template('home.html', 
                         gpu_available=gpu_available,
                         bozorth3_available=bozorth3_available)

@app.route("/matching")
def matching_page():
    return render_template('matching.html', 
                         gpu_available=gpu_available,
                         bozorth3_available=bozorth3_available)

@app.route("/benchmark")
def benchmark_page():
    return render_template('benchmark.html', 
                         gpu_available=gpu_available,
                         bozorth3_available=bozorth3_available)

@app.route('/extract', methods=['POST'])
def extract():
    """Extrai minúcias de uma imagem"""
    if 'file' not in request.files:
        return jsonify({'erro': 'Arquivo não enviado'}), 400
    
    file = request.files['file']
    image_path = Path(afis_cpu.temp_dir) / 'upload.png'
    file.save(image_path)

    try:
        xyt_path = afis_cpu.extract_minutiae(str(image_path))
        return jsonify({'xyt': xyt_path})
    except Exception as e:
        return jsonify({'erro': str(e)}), 500

def image_to_base64(image_path):
    """Converte imagem para base64"""
    try:
        with open(image_path, 'rb') as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Erro ao ler imagem {image_path}: {e}")
        return None

@app.route('/api/matching', methods=['POST'])
def api_matching():
    """
    Endpoint principal de matching
    Suporta 4 modos:
    - cpu: CPU tradicional (multiprocessing)
    - gpu: GPU simplificada (nossa implementação)
    - hybrid: GPU + CPU híbrido
    - bozorth3_gpu: Bozorth3 GPU (artigo)
    """
    data = request.json
    probe_xyt = data.get('probe_xyt')
    probe_image_b64 = data.get('probe_image')
    db_path = DATABASE_PATH
    mode = data.get('mode', 'cpu')
    cores = data.get('cores', 4)
    top_k = data.get('top_k', 5)
    threshold = data.get('threshold', 40)

    if not probe_xyt or not db_path:
        return jsonify({'erro': 'Parâmetros faltando'}), 400

    try:
        db_dir = Path(db_path)
        
        if not db_dir.exists():
            return jsonify({'erro': f'Diretório não encontrado: {db_path}'}), 404
        
        total_comparisons = len(list(db_dir.glob("*.xyt")))
        
        if total_comparisons == 0:
            return jsonify({'erro': 'Nenhum arquivo .xyt encontrado na base'}), 404
        
        print(f"\n🔍 Iniciando matching...")
        print(f"   Modo: {mode}")
        print(f"   Base: {db_path}")
        print(f"   Candidatos: {total_comparisons}")
        
        start = time.time()
        
        # ====================================================================
        # SELEÇÃO DO ENGINE
        # ====================================================================
        
        if mode == 'bozorth3_gpu':
            # NOVO: Bozorth3 GPU (artigo)
            if not bozorth3_gpu or not bozorth3_available:
                return jsonify({'erro': 'Bozorth3 GPU não disponível'}), 400
            
            print(f"   🚀 Usando Bozorth3 GPU (artigo)")
            
            # Bozorth3 usa score absoluto, não percentage
            # Ajusta threshold: se > 100, assume que é score absoluto
            bozorth3_threshold = threshold if threshold <= 100 else 40
            use_percentage = threshold <= 100
            
            matches = bozorth3_gpu.one_to_many_match(
                probe_xyt, 
                db_path, 
                top_k, 
                bozorth3_threshold,
                percentage_threshold=use_percentage
            )
            
            # Converte MatchResult para dict
            matches = [
                {
                    'candidate_id': m.candidate_id,
                    'score': m.score,
                    'percentage': m.percentage
                }
                for m in matches
            ]
        
        elif mode == 'gpu':
            # GPU simplificada
            if not afis_gpu or not gpu_available:
                return jsonify({'erro': 'GPU não disponível'}), 400
            
            matches = afis_gpu.one_many_match_gpu(probe_xyt, db_path, top_k, threshold)
        
        elif mode == 'hybrid':
            # Híbrido
            if not afis_gpu or not gpu_available:
                return jsonify({'erro': 'GPU não disponível'}), 400
            
            matches = afis_gpu.one_many_match_hybrid(probe_xyt, db_path, top_k, threshold)
        
        else:  # CPU
            original_cores = afis_cpu.num_cores
            afis_cpu.num_cores = cores
            matches = afis_cpu.one_many_match_cpu(probe_xyt, db_path, top_k, threshold)
            afis_cpu.num_cores = original_cores
            
            # Converte MatchResult para dict
            matches = [
                {
                    'candidate_id': m.candidate_id,
                    'score': m.score
                }
                for m in matches
            ]
        
        elapsed = time.time() - start
        
        print(f"✅ Matching concluído em {elapsed:.2f}s")
        print(f"   Matches encontrados: {len(matches)}")
        
        # ====================================================================
        # CARREGAMENTO DE IMAGENS
        # ====================================================================
        
        matches_with_images = []
        
        for idx, match in enumerate(matches):
            candidate_id = match['candidate_id'] if isinstance(match, dict) else match.candidate_id
            score = match['score'] if isinstance(match, dict) else match.score
            
            # Busca imagem PNG
            image_path = db_dir / f"{candidate_id}.png"
            
            if image_path.exists():
                img_base64 = image_to_base64(image_path)
                
                if img_base64:
                    matches_with_images.append({
                        'rank': idx + 1,
                        'candidate_id': candidate_id,
                        'score': float(score),
                        'image_base64': img_base64,
                        'percentage': match.get('percentage', None)
                    })
            else:
                matches_with_images.append({
                    'rank': idx + 1,
                    'candidate_id': candidate_id,
                    'score': float(score),
                    'image_base64': None,
                    'percentage': match.get('percentage', None)
                })
        
        return jsonify({
            'matches': matches_with_images,
            'total_comparisons': total_comparisons,
            'time': elapsed,
            'speed': total_comparisons / elapsed if elapsed > 0 else 0,
            'mode': mode
        })
        
    except Exception as e:
        import traceback
        print(f"❌ Erro no matching: {e}")
        print(traceback.format_exc())
        return jsonify({'erro': str(e)}), 500

@app.route('/api/benchmark', methods=['POST'])
def api_benchmark():
    """
    Endpoint de benchmark
    Testa CPU, GPU simplificada e Bozorth3 GPU
    """
    data = request.json
    probe_xyt = data.get('probe_xyt')
    db_path = data.get('db_path')

    try:
        db_dir = Path(db_path)
        total_comparisons = len(list(db_dir.glob("*.xyt")))
        results = {}
        
        # ====================================================================
        # CPU: 1, 4, 8 núcleos
        # ====================================================================
        cpu_times = {}
        for cores in [1, 4, 8]:
            print(f"\n🧪 Testando CPU ({cores} núcleos)...")
            original_cores = afis_cpu.num_cores
            afis_cpu.num_cores = cores
            
            start = time.time()
            afis_cpu.one_many_match_cpu(probe_xyt, db_path, top_k=10, threshold=40)
            elapsed = time.time() - start
            cpu_times[cores] = elapsed
            
            afis_cpu.num_cores = original_cores
            
            speedup = cpu_times[1] / elapsed if 1 in cpu_times else 1.0
            efficiency = (speedup / cores) * 100
            
            results[f'CPU_{cores}'] = {
                'time': elapsed,
                'speed': total_comparisons / elapsed,
                'speedup': speedup,
                'efficiency': efficiency,
                'type': 'CPU'
            }
        
        # ====================================================================
        # GPU Simplificada
        # ====================================================================
        if afis_gpu and gpu_available:
            print(f"\n🧪 Testando GPU (simplificada)...")
            start = time.time()
            afis_gpu.one_many_match_gpu(probe_xyt, db_path, top_k=10, threshold=40)
            gpu_time = time.time() - start
            
            results['GPU_Simple'] = {
                'time': gpu_time,
                'speed': total_comparisons / gpu_time,
                'speedup': cpu_times[1] / gpu_time,
                'efficiency': None,
                'type': 'GPU'
            }
            
            # Híbrido
            print(f"\n🧪 Testando Híbrido (GPU + CPU)...")
            start = time.time()
            afis_gpu.one_many_match_hybrid(probe_xyt, db_path, top_k=10, threshold=40)
            hybrid_time = time.time() - start
            
            results['GPU_Hybrid'] = {
                'time': hybrid_time,
                'speed': total_comparisons / hybrid_time,
                'speedup': cpu_times[1] / hybrid_time,
                'efficiency': None,
                'type': 'GPU'
            }
        
        # ====================================================================
        # Bozorth3 GPU (artigo)
        # ====================================================================
        if bozorth3_gpu and bozorth3_available:
            print(f"\n🧪 Testando Bozorth3 GPU (artigo)...")
            start = time.time()
            
            try:
                bozorth3_gpu.one_to_many_match(
                    probe_xyt, 
                    db_path, 
                    top_k=10, 
                    threshold=40,
                    percentage_threshold=True
                )
                bozorth3_time = time.time() - start
                
                results['Bozorth3_GPU'] = {
                    'time': bozorth3_time,
                    'speed': total_comparisons / bozorth3_time,
                    'speedup': cpu_times[1] / bozorth3_time,
                    'efficiency': None,
                    'type': 'GPU (Bozorth3)'
                }
                print(f"   ✅ Bozorth3 GPU: {bozorth3_time:.2f}s")
            except Exception as e:
                print(f"   ❌ Erro no Bozorth3 GPU: {e}")
                import traceback
                traceback.print_exc()
        
        return jsonify({'results': results})
    
    except Exception as e:
        import traceback
        print(f"❌ Erro no benchmark: {e}")
        print(traceback.format_exc())
        return jsonify({'erro': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 AFIS Web System com Bozorth3 GPU")
    print("="*60)
    print(f"   CPU: ✅ {afis_cpu.num_cores} núcleos")
    print(f"   GPU (simplificada): {'✅' if gpu_available else '❌'}")
    print(f"   Bozorth3 GPU: {'✅' if bozorth3_available else '❌'}")
    print(f"\n📍 Acesse: http://localhost:5000")
    print(f"\n⚠️  IMPORTANTE:")
    print(f"   - Certifique-se de ter CUDA Toolkit instalado")
    print(f"   - Instale: pip install numba")
    print("="*60)
    app.run(host='0.0.0.0', port=5000, debug=True)