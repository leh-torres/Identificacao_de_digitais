import subprocess
import numpy as np
import cupy as cp
from pathlib import Path
from typing import List, Tuple
import time
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import os

# Desabilita timeout do compilador CUDA
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUPY_GPU_MEMORY_SIZE'] = '0'  # Usar toda a memória disponível

# Aumenta timeouts CUDA
try:
    cp.cuda.runtime.deviceSetLimit(cp.cuda.runtime.cudaLimitDevRuntimeSyncDepth, 1)
    cp.cuda.runtime.deviceSetLimit(cp.cuda.runtime.cudaLimitMallocHeapSize, 256 * 1024 * 1024)
except:
    pass

@dataclass
class MinutiaeData:
    """Estrutura para armazenar dados de minúcias"""
    x: np.ndarray
    y: np.ndarray
    theta: np.ndarray
    quality: np.ndarray

class GPUAfisEngine:
    """
    Engine AFIS com aceleração GPU usando CuPy
    Implementa matching paralelo em GPU para comparações 1:N
    """
    
    def __init__(self, 
                 bozorth3_path='/home/leticia/nbis_build/bin/bozorth3',
                 use_hybrid=True,
                 gpu_batch_size=1000):
        """
        Args:
            bozorth3_path: Caminho para o binário bozorth3
            use_hybrid: Se True, usa abordagem híbrida CPU+GPU
            gpu_batch_size: Número de comparações por batch na GPU
        """
        self.bozorth3_path = bozorth3_path
        self.use_hybrid = use_hybrid
        self.gpu_batch_size = gpu_batch_size
        
        # Verifica disponibilidade da GPU
        try:
            self.device = cp.cuda.Device(0)
            self.gpu_available = True
            print(f"✅ GPU detectada: {self.device.compute_capability}")
        except:
            self.gpu_available = False
            print("⚠️ GPU não disponível, usando CPU")
    
    def parse_xyt_file(self, xyt_path: str) -> MinutiaeData:
        """
        Parse arquivo .xyt e retorna dados estruturados
        Formato XYT: x y theta quality
        """
        data = np.loadtxt(xyt_path, dtype=float)
        
        if data.size == 0:
            raise ValueError(f"Arquivo vazio: {xyt_path}")
        
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        return MinutiaeData(
            x=data[:, 0].astype(np.float32),
            y=data[:, 1].astype(np.float32),
            theta=data[:, 2].astype(np.float32),
            quality=data[:, 3].astype(np.float32) if data.shape[1] > 3 else np.ones(len(data), dtype=np.float32)
        )
    
    def load_database_to_gpu(self, db_path: str) -> Tuple[List[str], cp.ndarray]:
        """
        Carrega toda a base de dados para a memória da GPU
        Retorna lista de IDs e tensor com todas as minúcias
        """
        db_dir = Path(db_path)
        xyt_files = sorted(list(db_dir.glob('*.xyt')))
        
        if not xyt_files:
            raise ValueError(f"Nenhum arquivo .xyt encontrado em {db_path}")
        
        print(f"📥 Carregando {len(xyt_files)} impressões para GPU...")
        
        ids = []
        all_minutiae = []
        max_minutiae = 0
        
        # Primeira passagem: determina tamanho máximo
        minutiae_list = []
        for xyt_file in xyt_files:
            minutiae = self.parse_xyt_file(str(xyt_file))
            minutiae_list.append(minutiae)
            max_minutiae = max(max_minutiae, len(minutiae.x))
            ids.append(xyt_file.stem)
        
        # Segunda passagem: cria tensor padronizado
        # Shape: (num_prints, max_minutiae, 4) -> [x, y, theta, quality]
        db_tensor = np.zeros((len(xyt_files), max_minutiae, 4), dtype=np.float32)
        
        for i, minutiae in enumerate(minutiae_list):
            n = len(minutiae.x)
            db_tensor[i, :n, 0] = minutiae.x
            db_tensor[i, :n, 1] = minutiae.y
            db_tensor[i, :n, 2] = minutiae.theta
            db_tensor[i, :n, 3] = minutiae.quality
        
        # Transfere para GPU
        db_tensor_gpu = cp.asarray(db_tensor)
        
        print(f"✅ Base carregada: {db_tensor_gpu.shape}")
        print(f"   Memória GPU: {db_tensor_gpu.nbytes / 1024**2:.2f} MB")
        
        return ids, db_tensor_gpu
    
    def compute_similarity_gpu(self, 
                               probe_minutiae: MinutiaeData,
                               db_tensor_gpu: cp.ndarray) -> cp.ndarray:
        """
        Calcula similaridade entre probe e todos os candidatos na GPU
        Implementa algoritmo simplificado de matching baseado em distância
        
        Args:
            probe_minutiae: Minúcias da impressão probe
            db_tensor_gpu: Tensor com todas as impressões da base
            
        Returns:
            Array com scores de similaridade
        """
        try:
            # Converte probe para GPU - usando método mais eficiente
            n_db = db_tensor_gpu.shape[0]
            max_minutiae = db_tensor_gpu.shape[1]
            n_probe = len(probe_minutiae.x)
            
            # Cria tensor do probe com alocação pré-definida
            probe_tensor = cp.zeros((max_minutiae, 4), dtype=cp.float32)
            probe_tensor[:n_probe, 0] = cp.asarray(probe_minutiae.x, dtype=cp.float32)
            probe_tensor[:n_probe, 1] = cp.asarray(probe_minutiae.y, dtype=cp.float32)
            probe_tensor[:n_probe, 2] = cp.asarray(probe_minutiae.theta, dtype=cp.float32)
            probe_tensor[:n_probe, 3] = cp.asarray(probe_minutiae.quality, dtype=cp.float32)
            
            # Parâmetros de matching
            spatial_threshold = 10.0  # pixels
            angle_threshold = 15.0    # graus
            
            # Cria array de scores
            scores = cp.zeros(n_db, dtype=cp.float32)
            
            # Processa em chunks para evitar compilação pesada
            chunk_size = 100
            for db_idx in range(0, n_db, chunk_size):
                end_idx = min(db_idx + chunk_size, n_db)
                db_chunk = db_tensor_gpu[db_idx:end_idx]  # (chunk_size, max_minutiae, 4)
                
                # Calcula distâncias de forma vetorizada mas sem kernels complexos
                probe_x = probe_tensor[:, 0]  # (max_minutiae,)
                probe_y = probe_tensor[:, 1]
                probe_theta = probe_tensor[:, 2]
                probe_quality = probe_tensor[:, 3]
                
                db_x = db_chunk[:, :, 0]  # (chunk_size, max_minutiae)
                db_y = db_chunk[:, :, 1]
                db_theta = db_chunk[:, :, 2]
                db_quality = db_chunk[:, :, 3]
                
                # Distância euclidiana
                dx = probe_x - db_x  # Broadcasting: (chunk_size, max_minutiae)
                dy = probe_y - db_y
                spatial_dist = cp.sqrt(dx**2 + dy**2)
                
                # Diferença angular
                angle_diff = cp.abs(probe_theta - db_theta)
                angle_diff = cp.minimum(angle_diff, 360.0 - angle_diff)
                
                # Matching: minúcias que satisfazem os critérios
                matches = (spatial_dist < spatial_threshold) & (angle_diff < angle_threshold)
                
                # Score ponderado
                quality_prod = probe_quality * db_quality  # Broadcasting
                weighted = matches.astype(cp.float32) * quality_prod
                
                # Soma dos matches por candidato
                chunk_scores = cp.sum(weighted, axis=1)
                
                # Normaliza
                n_minutiae = cp.sum(db_quality > 0, axis=1)
                chunk_scores = chunk_scores / cp.maximum(n_minutiae.astype(cp.float32), 1.0)
                chunk_scores = chunk_scores * 100.0  # Escala para 0-400 (similar ao bozorth3)
                
                scores[db_idx:end_idx] = chunk_scores
                
                # Libera memória do chunk
                del db_chunk, dx, dy, spatial_dist, angle_diff, matches, weighted
                cp.get_default_memory_pool().free_all_blocks()
            
            return scores
            
        except cp.cuda.runtime.CUDADriverError as e:
            if "LAUNCH_TIMEOUT" in str(e):
                print(f"⚠️  CUDA timeout - tentando com batch menor...")
                # Recursive retry com batch muito menor
                return self.compute_similarity_gpu_cpu_fallback(probe_minutiae, db_tensor_gpu)
            raise
    
    def compute_similarity_gpu_cpu_fallback(self,
                                           probe_minutiae: MinutiaeData,
                                           db_tensor_gpu: cp.ndarray) -> np.ndarray:
        """
        Fallback para CPU se GPU der timeout
        """
        print("🔄 Usando fallback CPU...")
        db_tensor = cp.asnumpy(db_tensor_gpu)
        
        spatial_threshold = 10.0
        angle_threshold = 15.0
        
        n_db = db_tensor.shape[0]
        scores = np.zeros(n_db, dtype=np.float32)
        
        for i in range(n_db):
            db_minutiae = db_tensor[i]
            
            # Remove padding (qualidade = 0)
            valid_idx = db_minutiae[:, 3] > 0
            probe_valid_idx = probe_minutiae.quality > 0
            
            if not (valid_idx.any() and probe_valid_idx.any()):
                continue
            
            db_valid = db_minutiae[valid_idx]
            
            probe_x = probe_minutiae.x[probe_valid_idx]
            probe_y = probe_minutiae.y[probe_valid_idx]
            probe_theta = probe_minutiae.theta[probe_valid_idx]
            probe_quality = probe_minutiae.quality[probe_valid_idx]
            
            for j, (px, py, pt, pq) in enumerate(zip(probe_x, probe_y, probe_theta, probe_quality)):
                dx = px - db_valid[:, 0]
                dy = py - db_valid[:, 1]
                dist = np.sqrt(dx**2 + dy**2)
                
                angle_diff = np.abs(pt - db_valid[:, 2])
                angle_diff = np.minimum(angle_diff, 360 - angle_diff)
                
                matches = (dist < spatial_threshold) & (angle_diff < angle_threshold)
                scores[i] += np.sum(matches * pq * db_valid[matches, 3]) if matches.any() else 0
            
            scores[i] = scores[i] / len(db_valid) * 100
        
        return scores
    
    def one_many_match_gpu(self, 
                           probe_path: str,
                           db_path: str,
                           top_k: int = 10,
                           threshold: float = 40) -> List[dict]:
        """
        Busca 1:N acelerada por GPU
        
        Returns:
            Lista de matches ordenados por score
        """
        if not self.gpu_available:
            raise RuntimeError("GPU não disponível")
        
        print(f"🚀 Iniciando matching GPU...")
        start_total = time.time()
        
        # Carrega probe
        probe_minutiae = self.parse_xyt_file(probe_path)
        
        # Carrega base para GPU
        start_load = time.time()
        ids, db_tensor_gpu = self.load_database_to_gpu(db_path)
        load_time = time.time() - start_load
        print(f"⏱️  Tempo de carregamento: {load_time:.3f}s")
        
        # Processa em batches se necessário
        start_compute = time.time()
        n_candidates = len(ids)
        
        if n_candidates <= self.gpu_batch_size:
            # Processa tudo de uma vez
            scores_gpu = self.compute_similarity_gpu(probe_minutiae, db_tensor_gpu)
            # Verifica se é numpy array (fallback CPU) ou CuPy array
            if isinstance(scores_gpu, np.ndarray):
                scores = scores_gpu
            else:
                scores = cp.asnumpy(scores_gpu)
        else:
            # Processa em batches
            scores = np.zeros(n_candidates, dtype=np.float32)
            for i in range(0, n_candidates, self.gpu_batch_size):
                end_idx = min(i + self.gpu_batch_size, n_candidates)
                batch_tensor = db_tensor_gpu[i:end_idx]
                batch_scores = self.compute_similarity_gpu(probe_minutiae, batch_tensor)
                
                # Verifica tipo de retorno
                if isinstance(batch_scores, np.ndarray):
                    scores[i:end_idx] = batch_scores
                else:
                    scores[i:end_idx] = cp.asnumpy(batch_scores)
        
        compute_time = time.time() - start_compute
        total_time = time.time() - start_total
        
        # Prepara resultados
        results = []
        for i, (candidate_id, score) in enumerate(zip(ids, scores)):
            if score >= threshold:
                results.append({
                    'probe_id': Path(probe_path).stem,
                    'candidate_id': candidate_id,
                    'score': float(score),
                    'rank': 0  # Será atualizado depois
                })
        
        # Ordena e pega top K
        results.sort(key=lambda x: x['score'], reverse=True)
        results = results[:top_k]
        
        # Atualiza ranks
        for rank, result in enumerate(results, 1):
            result['rank'] = rank
        
        # Estatísticas
        print(f"\n✅ Matching concluído!")
        print(f"   Total: {total_time:.3f}s")
        print(f"   Carregamento: {load_time:.3f}s")
        print(f"   Computação: {compute_time:.3f}s")
        print(f"   Velocidade: {n_candidates/compute_time:.1f} comp/seg")
        print(f"   Matches encontrados: {len(results)}")
        
        return results
    
    def one_many_match_hybrid(self,
                             probe_path: str,
                             db_path: str,
                             top_k: int = 10,
                             threshold: float = 40,
                             pre_filter_k: int = 50) -> List[dict]:
        """
        Abordagem híbrida: GPU para pré-filtro rápido + CPU para refinamento
        
        Args:
            pre_filter_k: Quantos candidatos passar para refinamento CPU
        """
        print(f"🔄 Matching híbrido GPU+CPU...")
        
        # Fase 1: Pré-filtro rápido na GPU
        start_gpu = time.time()
        gpu_results = self.one_many_match_gpu(
            probe_path, 
            db_path, 
            top_k=pre_filter_k,
            threshold=0  # Sem threshold no pré-filtro
        )
        gpu_time = time.time() - start_gpu
        
        print(f"\n📊 Fase GPU: {len(gpu_results)} candidatos selecionados")
        
        # Fase 2: Refinamento com Bozorth3 (CPU)
        print(f"🔬 Refinando top {pre_filter_k} com Bozorth3...")
        start_cpu = time.time()
        
        db_dir = Path(db_path)
        refined_results = []
        
        for result in gpu_results[:pre_filter_k]:
            candidate_path = db_dir / f"{result['candidate_id']}.xyt"
            
            # Executa Bozorth3
            cmd = [self.bozorth3_path, probe_path, str(candidate_path)]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            
            if proc.returncode == 0:
                try:
                    bozorth_score = int(proc.stdout.strip())
                    refined_results.append({
                        'probe_id': result['probe_id'],
                        'candidate_id': result['candidate_id'],
                        'score': bozorth_score,
                        'gpu_score': result['score']
                    })
                except:
                    pass
        
        cpu_time = time.time() - start_cpu
        total_time = gpu_time + cpu_time
        
        # Ordena por score Bozorth3
        refined_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Filtra por threshold e pega top K
        filtered = [r for r in refined_results if r['score'] >= threshold]
        final_results = filtered[:top_k]
        
        # Adiciona ranks
        for rank, result in enumerate(final_results, 1):
            result['rank'] = rank
        
        print(f"\n✅ Matching híbrido concluído!")
        print(f"   Fase GPU: {gpu_time:.3f}s")
        print(f"   Fase CPU: {cpu_time:.3f}s")
        print(f"   Total: {total_time:.3f}s")
        print(f"   Matches finais: {len(final_results)}")
        
        return final_results
    
    def benchmark_gpu_vs_cpu(self,
                            probe_path: str,
                            db_path: str,
                            cpu_cores: int = 4) -> dict:
        """
        Compara performance GPU vs CPU
        """
        from afis_engine import AfisEngine
        
        print("=" * 60)
        print("BENCHMARK: GPU vs CPU")
        print("=" * 60)
        
        db_dir = Path(db_path)
        n_candidates = len(list(db_dir.glob('*.xyt')))
        
        results = {}
        
        # Teste 1: CPU (baseline)
        print(f"\n1️⃣ CPU ({cpu_cores} núcleos)...")
        cpu_engine = AfisEngine(
            bozorth3_path=self.bozorth3_path,
            num_cores=cpu_cores
        )
        
        start = time.time()
        cpu_matches = cpu_engine.one_many_match_cpu(
            probe_path, db_path, top_k=10, threshold=40
        )
        cpu_time = time.time() - start
        
        results['cpu'] = {
            'time': cpu_time,
            'speed': n_candidates / cpu_time,
            'matches': len(cpu_matches),
            'top_score': cpu_matches[0].score if cpu_matches else 0
        }
        
        # Teste 2: GPU
        if self.gpu_available:
            print(f"\n2️⃣ GPU...")
            start = time.time()
            gpu_matches = self.one_many_match_gpu(
                probe_path, db_path, top_k=10, threshold=40
            )
            gpu_time = time.time() - start
            
            results['gpu'] = {
                'time': gpu_time,
                'speed': n_candidates / gpu_time,
                'matches': len(gpu_matches),
                'top_score': gpu_matches[0]['score'] if gpu_matches else 0,
                'speedup': cpu_time / gpu_time
            }
        
        # Teste 3: Híbrido
        if self.gpu_available and self.use_hybrid:
            print(f"\n3️⃣ Híbrido (GPU + CPU)...")
            start = time.time()
            hybrid_matches = self.one_many_match_hybrid(
                probe_path, db_path, top_k=10, threshold=40, pre_filter_k=50
            )
            hybrid_time = time.time() - start
            
            results['hybrid'] = {
                'time': hybrid_time,
                'speed': n_candidates / hybrid_time,
                'matches': len(hybrid_matches),
                'top_score': hybrid_matches[0]['score'] if hybrid_matches else 0,
                'speedup': cpu_time / hybrid_time
            }
        
        # Sumário
        print("\n" + "=" * 60)
        print("SUMÁRIO")
        print("=" * 60)
        
        for mode, data in results.items():
            print(f"\n{mode.upper()}:")
            print(f"  Tempo: {data['time']:.3f}s")
            print(f"  Velocidade: {data['speed']:.1f} comp/seg")
            if 'speedup' in data:
                print(f"  Speedup: {data['speedup']:.2f}x")
            print(f"  Matches: {data['matches']}")
            print(f"  Top score: {data['top_score']:.1f}")
        
        return results


if __name__ == "__main__":
    # Exemplo de uso
    engine = GPUAfisEngine(
        bozorth3_path='/home/leticia/nbis_build/bin/bozorth3',
        use_hybrid=True,
        gpu_batch_size=1000
    )
    
    probe = "/home/leticia/documentos/sspce/data/db_minutias_nist/0666_S_05_F_R.xyt"
    db = "/home/leticia/documentos/sspce/data/db_minutias_nist/"
    
    # Teste 1: GPU puro
    # results = engine.one_many_match_gpu(probe, db, top_k=10)
    
    # Teste 2: Híbrido
    # results = engine.one_many_match_hybrid(probe, db, top_k=10)
    
    # Teste 3: Benchmark completo
    benchmark_results = engine.benchmark_gpu_vs_cpu(probe, db, cpu_cores=4)