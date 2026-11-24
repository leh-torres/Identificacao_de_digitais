import subprocess
import shutil
import uuid
import time
import numpy as np

from pathlib import Path
from typing import List, Tuple, Optional
import multiprocessing
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor

# Importação condicional do CuPy
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

@dataclass
class MatchResult:
    probe_id: str
    candidate_id: str
    score: int
    time_ms: float
    gpu_score: Optional[float] = None  # Score do pré-filtro GPU

class AfisEngine:
    """
    Engine AFIS usando mindtct para extração e bozorth3 para matching
    Suporta paralelização em CPU (multiprocessing) e GPU (via CUDA/CuPy)
    """

    def __init__(self,
                 mindtct_path='/app/lib/nbis_build/bin/mindtct',
                 bozorth3_path='/app/lib/nbis_build/bin/bozorth3',
                 temp_dir='/temp',
                 num_cores=4,
                 use_gpu=False,
                 gpu_batch_size=1000):
        self.mindtct_path = mindtct_path
        self.bozorth3_path = bozorth3_path
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.num_cores = num_cores
        self.use_gpu = use_gpu
        self.gpu_batch_size = gpu_batch_size

        # Verifica se os binários existem
        if not Path(mindtct_path).exists():
            raise FileNotFoundError(f'Mindtct não encontrado em {mindtct_path}')
        
        if not Path(bozorth3_path).exists():
            raise FileNotFoundError(f'Bozorth3 não encontrado em {bozorth3_path}')
        
        # Verifica GPU
        self.gpu_available = False
        if use_gpu:
            if not CUPY_AVAILABLE:
                print("⚠️ CuPy não instalado. Instale com: pip install cupy-cuda11x")
            else:
                try:
                    self.device = cp.cuda.Device(0)
                    self.gpu_available = True
                    print(f"✅ GPU detectada: {self.device.compute_capability}")
                    print(f"   Memória disponível: {self.device.mem_info[0] / 1024**3:.2f} GB")
                except Exception as e:
                    print(f"⚠️ Erro ao inicializar GPU: {e}")
    
    def clear_temp(self):
        """Limpar diretório temporário"""
        for item in self.temp_dir.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
    
    def extract_minutiae(self, image_path: str) -> str:
        """
        Executa mindtct sobre um PNG e gera arquivo .xyt
        Retorna o caminho do arquivo .xyt gerado
        """
        uid = str(uuid.uuid4())
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        output_base = self.temp_dir / uid

        cmd = [
            self.mindtct_path,
            image_path,
            str(output_base)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f'Erro no mindtct: {result.stderr}')

        xyt_file = output_base.with_suffix('.xyt')

        if not xyt_file.exists():
            raise RuntimeError('Falha na extração de minúcias')
        
        for f in self.temp_dir.iterdir():
            if f.stem == uid and f.suffix != '.xyt':
                f.unlink()

        return str(xyt_file)
    
    def one_one_match(self, f1_path: str, f2_path: str) -> MatchResult:
        """
        Compara duas impressões digitais usando bozorth3
        Retorna score de similaridade (quanto maior, mais similar)
        """
        start = time.time()

        cmd = [
            self.bozorth3_path,
            f1_path,
            f2_path
        ]

        result = subprocess.run(
            cmd, 
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if result.returncode != 0:
            raise RuntimeError(f'Erro no bozorth3: {result.stderr}')
        
        try:
            score = int(result.stdout.strip())
        except ValueError:
            raise RuntimeError(f'Score inválido: {result.stdout}')
        
        one_one_time = (time.time() - start) * 1000

        return MatchResult(
            probe_id=Path(f1_path).stem,
            candidate_id=Path(f2_path).stem,
            score=score,
            time_ms=one_one_time
        )
    
    def _match_worker(self, args: Tuple[str, str, str]) -> MatchResult:
        """Worker para processamento paralelo"""
        probe_path, candidate_path, bozorth3_path = args
        
        cmd = [bozorth3_path, probe_path, candidate_path]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        score = 0
        if result.returncode == 0:
            try:
                score = int(result.stdout.strip())
            except:
                score = 0

        return MatchResult(
            probe_id=Path(probe_path).stem,
            candidate_id=Path(candidate_path).stem,
            score=score,
            time_ms=0
        )
    
    def one_many_match_cpu(self, probe_path: str, db_path: str, 
                          top_k: int = 10, threshold: int = 40) -> List[MatchResult]:
        """
        Busca 1:N usando paralelização em CPU
        """
        db_dir = Path(db_path)

        if not db_dir.exists():
            raise FileNotFoundError(f'Diretório da base não encontrado: {db_path}')
        
        candidates = list(db_dir.glob('*.xyt'))

        if not candidates:
            raise ValueError(f'Nenhum arquivo .xyt encontrado em {db_path}')
        
        print(f"Comparando contra {len(candidates)} candidatos usando {self.num_cores} núcleos...")

        start = time.time()

        tasks = [(probe_path, str(c), self.bozorth3_path) for c in candidates] 

        with ProcessPoolExecutor(max_workers=self.num_cores) as executor:
            results = list(executor.map(self._match_worker, tasks))

        one_many_cpu_time = time.time() - start

        filtered = [r for r in results if r.score >= threshold]
        sorted_results = sorted(filtered, key=lambda x: x.score, reverse=True)[:top_k]
        
        for r in sorted_results:
            r.time_ms = one_many_cpu_time * 1000 / len(candidates)
        
        print(f"Matching concluído em {one_many_cpu_time:.2f}s")
        print(f"Velocidade: {len(candidates)/one_many_cpu_time:.1f} comparações/seg")
        
        return sorted_results
    
    def _parse_xyt_file(self, xyt_path: str) -> dict:
        """Parse arquivo .xyt"""
        data = np.loadtxt(xyt_path, dtype=float)
        
        if data.size == 0:
            raise ValueError(f"Arquivo vazio: {xyt_path}")
        
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        return {
            'x': data[:, 0].astype(np.float32),
            'y': data[:, 1].astype(np.float32),
            'theta': data[:, 2].astype(np.float32),
            'quality': data[:, 3].astype(np.float32) if data.shape[1] > 3 else np.ones(len(data), dtype=np.float32)
        }
    
    def _load_database_to_gpu(self, db_path: str) -> Tuple[List[str], cp.ndarray]:
        """Carrega base para GPU"""
        db_dir = Path(db_path)
        xyt_files = sorted(list(db_dir.glob('*.xyt')))
        
        print(f"📥 Carregando {len(xyt_files)} impressões para GPU...")
        
        ids = []
        minutiae_list = []
        max_minutiae = 0
        
        for xyt_file in xyt_files:
            minutiae = self._parse_xyt_file(str(xyt_file))
            minutiae_list.append(minutiae)
            max_minutiae = max(max_minutiae, len(minutiae['x']))
            ids.append(xyt_file.stem)
        
        # Cria tensor padronizado
        db_tensor = np.zeros((len(xyt_files), max_minutiae, 4), dtype=np.float32)
        
        for i, minutiae in enumerate(minutiae_list):
            n = len(minutiae['x'])
            db_tensor[i, :n, 0] = minutiae['x']
            db_tensor[i, :n, 1] = minutiae['y']
            db_tensor[i, :n, 2] = minutiae['theta']
            db_tensor[i, :n, 3] = minutiae['quality']
        
        db_tensor_gpu = cp.asarray(db_tensor)
        
        print(f"✅ Base carregada: {db_tensor_gpu.shape}")
        print(f"   Memória GPU: {db_tensor_gpu.nbytes / 1024**2:.2f} MB")
        
        return ids, db_tensor_gpu
    
    def _compute_similarity_gpu(self, probe_minutiae: dict, db_tensor_gpu: cp.ndarray) -> cp.ndarray:
        """Calcula similaridade na GPU"""
        probe_tensor = cp.zeros((1, db_tensor_gpu.shape[1], 4), dtype=cp.float32)
        n_probe = len(probe_minutiae['x'])
        probe_tensor[0, :n_probe, 0] = cp.asarray(probe_minutiae['x'])
        probe_tensor[0, :n_probe, 1] = cp.asarray(probe_minutiae['y'])
        probe_tensor[0, :n_probe, 2] = cp.asarray(probe_minutiae['theta'])
        probe_tensor[0, :n_probe, 3] = cp.asarray(probe_minutiae['quality'])
        
        # Matching parameters
        spatial_threshold = 10.0
        angle_threshold = 15.0
        
        probe_expanded = probe_tensor[:, :, cp.newaxis, :]
        db_expanded = db_tensor_gpu[:, cp.newaxis, :, :]
        
        spatial_diff = probe_expanded[:, :, :, :2] - db_expanded[:, :, :, :2]
        spatial_dist = cp.sqrt(cp.sum(spatial_diff ** 2, axis=-1))
        
        angle_diff = cp.abs(probe_expanded[:, :, :, 2] - db_expanded[:, :, :, 2])
        angle_diff = cp.minimum(angle_diff, 360 - angle_diff)
        
        matches = (spatial_dist < spatial_threshold) & (angle_diff < angle_threshold)
        
        quality_product = probe_expanded[:, :, :, 3] * db_expanded[:, :, :, 3]
        weighted_matches = matches.astype(cp.float32) * quality_product
        
        scores = cp.sum(weighted_matches, axis=(1, 2))[0]
        n_minutiae = cp.sum(db_tensor_gpu[:, :, 3] > 0, axis=1)
        scores = scores / cp.maximum(n_minutiae, 1) * 100
        
        return scores
    
    def one_many_match_gpu(self, probe_path: str, db_path: str,
                          top_k: int = 10, threshold: int = 40,
                          hybrid: bool = True, pre_filter_k: int = 50) -> List[MatchResult]:
        """
        Busca 1:N usando GPU
        
        Args:
            hybrid: Se True, usa GPU para pré-filtro e CPU para refinamento
            pre_filter_k: Quantos candidatos refinar com Bozorth3 (modo híbrido)
        """
        if not self.gpu_available:
            print("⚠️ GPU não disponível, usando CPU")
            return self.one_many_match_cpu(probe_path, db_path, top_k, threshold)
        
        print(f"🚀 Iniciando matching GPU {'(híbrido)' if hybrid else ''}...")
        start_total = time.time()
        
        # Carrega probe e base
        probe_minutiae = self._parse_xyt_file(probe_path)
        start_load = time.time()
        ids, db_tensor_gpu = self._load_database_to_gpu(db_path)
        load_time = time.time() - start_load
        
        # Computação GPU
        start_compute = time.time()
        n_candidates = len(ids)
        
        if n_candidates <= self.gpu_batch_size:
            scores_gpu = self._compute_similarity_gpu(probe_minutiae, db_tensor_gpu)
            scores = cp.asnumpy(scores_gpu)
        else:
            scores = np.zeros(n_candidates, dtype=np.float32)
            for i in range(0, n_candidates, self.gpu_batch_size):
                end_idx = min(i + self.gpu_batch_size, n_candidates)
                batch_scores = self._compute_similarity_gpu(probe_minutiae, db_tensor_gpu[i:end_idx])
                scores[i:end_idx] = cp.asnumpy(batch_scores)
        
        compute_time = time.time() - start_compute
        
        # Prepara resultados iniciais
        gpu_results = []
        for candidate_id, score in zip(ids, scores):
            gpu_results.append({
                'candidate_id': candidate_id,
                'score': float(score)
            })
        
        gpu_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Modo híbrido: refinamento com Bozorth3
        if hybrid:
            print(f"\n🔬 Refinando top {pre_filter_k} com Bozorth3...")
            start_refine = time.time()
            
            db_dir = Path(db_path)
            final_results = []
            
            for result in gpu_results[:pre_filter_k]:
                candidate_path = db_dir / f"{result['candidate_id']}.xyt"
                
                cmd = [self.bozorth3_path, probe_path, str(candidate_path)]
                proc = subprocess.run(cmd, capture_output=True, text=True)
                
                if proc.returncode == 0:
                    try:
                        bozorth_score = int(proc.stdout.strip())
                        if bozorth_score >= threshold:
                            final_results.append(MatchResult(
                                probe_id=Path(probe_path).stem,
                                candidate_id=result['candidate_id'],
                                score=bozorth_score,
                                time_ms=0,
                                gpu_score=result['score']
                            ))
                    except:
                        pass
            
            refine_time = time.time() - start_refine
            final_results.sort(key=lambda x: x.score, reverse=True)
            final_results = final_results[:top_k]
            
            total_time = time.time() - start_total
            print(f"\n✅ Matching híbrido concluído!")
            print(f"   Carregamento: {load_time:.3f}s")
            print(f"   GPU: {compute_time:.3f}s")
            print(f"   Refinamento: {refine_time:.3f}s")
            print(f"   Total: {total_time:.3f}s")
            print(f"   Matches: {len(final_results)}")
            
            return final_results
        
        # Modo GPU puro
        else:
            results = []
            for result in gpu_results:
                if result['score'] >= threshold:
                    results.append(MatchResult(
                        probe_id=Path(probe_path).stem,
                        candidate_id=result['candidate_id'],
                        score=int(result['score']),
                        time_ms=0
                    ))
            
            results = results[:top_k]
            total_time = time.time() - start_total
            
            print(f"\n✅ Matching GPU concluído!")
            print(f"   Total: {total_time:.3f}s")
            print(f"   Velocidade: {n_candidates/compute_time:.1f} comp/seg")
            
            return results
    
    def benchmark(self, probe_path: str, db_path: str, 
                 cores_list: List[int] = [1, 4, 8],
                 test_gpu: bool = True):
        """
        Executa benchmark com diferentes números de núcleos e GPU
        """
        print("=" * 60)
        print("BENCHMARK DE PERFORMANCE")
        print("=" * 60)
        
        db_dir = Path(db_path)
        num_candidates = len(list(db_dir.glob('*.xyt')))

        print(f"\nBase de dados: {num_candidates} impressões digitais")
        print(f"Probe: {Path(probe_path).name}\n")

        results = {}
        
        # Testes CPU
        for cores in cores_list:
            print(f"--- Testando com {cores} núcleo(s) ---")
            original_cores = self.num_cores
            self.num_cores = cores
            
            start = time.time()
            matches = self.one_many_match_cpu(probe_path, db_path, top_k=5)
            elapsed = time.time() - start
            
            results[f'cpu_{cores}'] = {
                'time': elapsed,
                'speed': num_candidates / elapsed,
                'matches': matches
            }
            
            print(f"Tempo: {elapsed:.2f}s")
            print(f"Velocidade: {num_candidates/elapsed:.1f} comparações/seg\n")
            
            self.num_cores = original_cores
        
        # Teste GPU
        if test_gpu and self.gpu_available:
            print("--- Testando GPU (puro) ---")
            start = time.time()
            gpu_matches = self.one_many_match_gpu(
                probe_path, db_path, top_k=5, hybrid=False
            )
            gpu_time = time.time() - start
            
            results['gpu_pure'] = {
                'time': gpu_time,
                'speed': num_candidates / gpu_time,
                'matches': gpu_matches
            }
            
            print("--- Testando GPU (híbrido) ---")
            start = time.time()
            hybrid_matches = self.one_many_match_gpu(
                probe_path, db_path, top_k=5, hybrid=True
            )
            hybrid_time = time.time() - start
            
            results['gpu_hybrid'] = {
                'time': hybrid_time,
                'speed': num_candidates / hybrid_time,
                'matches': hybrid_matches
            }

        # Análise de speedup
        print("\n" + "=" * 60)
        print("ANÁLISE DE SPEEDUP")
        print("=" * 60)
        
        baseline = results.get('cpu_1', results[list(results.keys())[0]])['time']
        
        for name, result in results.items():
            speedup = baseline / result['time']
            print(f"{name}: {result['time']:.2f}s - Speedup: {speedup:.2f}x - "
                  f"{result['speed']:.1f} comp/seg")
        
        return results


if __name__ == "__main__":
    # Configuração com GPU
    afis = AfisEngine(
        mindtct_path='/app/lib/nbis_build/bin/mindtct',
        bozorth3_path='/app/lib/nbis_build/bin/bozorth3',
        temp_dir='/temp',
        num_cores=4,
        use_gpu=True  # Habilita GPU
    )
    
    # Exemplo: Busca com GPU
    # matches = afis.one_many_match_gpu(
    #     probe_path="/path/to/probe.xyt",
    #     db_path="/path/to/database/",
    #     top_k=10,
    #     hybrid=True  # Usa modo híbrido GPU+CPU
    # )
    
    # Exemplo: Benchmark completo
    results = afis.benchmark(
         probe_path="/home/leticia/documentos/sspce/data/db_minutiae_nist/0997_F_02_M_L.xyt",
         db_path="/home/leticia/documentos/sspce/data/db_minutiae_nist",
         cores_list=[1, 4, 8],
         test_gpu=True
    )