"""
Implementação GPU do Bozorth3 seguindo o artigo:
"Fingerprint Matching Using Bozorth3 Algorithm and Parallel Computation on NVIDIA CUDA"
Supatmi & Sumitra (2020)

Abordagem:
1. Envia minúcias {x,y,theta} para Shared Memory GPU
2. Calcula distâncias e ângulos em paralelo (GPU)
3. Retorna arrays para CPU
4. CPU faz comparação final e cálculo de score
"""

import numpy as np
from numba import cuda, float32, int32
import math
import time
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass

# Verifica disponibilidade CUDA
try:
    cuda.select_device(0)
    CUDA_AVAILABLE = True
    print("✅ CUDA disponível")
except:
    CUDA_AVAILABLE = False
    print("❌ CUDA não disponível")

@dataclass
class MinutiaeData:
    """Estrutura para armazenar minúcias"""
    x: np.ndarray
    y: np.ndarray
    theta: np.ndarray
    quality: np.ndarray

@dataclass
class MatchResult:
    """Resultado de matching"""
    probe_id: str
    candidate_id: str
    score: int
    percentage: float
    time_ms: float

# ============================================================================
# KERNELS CUDA (GPU)
# ============================================================================

@cuda.jit
def calculate_distances_angles_kernel(probe_x, probe_y, probe_theta,
                                     gallery_x, gallery_y, gallery_theta,
                                     distances, angles, n_probe, n_gallery):
    """
    Kernel CUDA para calcular distâncias e ângulos entre minúcias
    Cada thread processa uma combinação probe-gallery
    
    Similar ao artigo: "Each thread will do the process of parallel computing"
    """
    # Shared memory para probe (acesso rápido)
    shared_probe_x = cuda.shared.array(shape=(128,), dtype=float32)
    shared_probe_y = cuda.shared.array(shape=(128,), dtype=float32)
    shared_probe_theta = cuda.shared.array(shape=(128,), dtype=float32)
    
    # Identificação do thread
    tx = cuda.threadIdx.x  # Thread dentro do block
    bx = cuda.blockIdx.x   # Block ID
    bw = cuda.blockDim.x   # Block width (threads por block)
    
    idx = bx * bw + tx  # Índice global do thread
    
    # Copia probe para shared memory (cooperativo entre threads do block)
    if tx < n_probe and tx < 128:
        shared_probe_x[tx] = probe_x[tx]
        shared_probe_y[tx] = probe_y[tx]
        shared_probe_theta[tx] = probe_theta[tx]
    
    cuda.syncthreads()  # Sincroniza threads do block
    
    # Cada thread processa múltiplas combinações
    if idx < n_probe * n_gallery:
        i = idx // n_gallery  # Índice probe
        j = idx % n_gallery   # Índice gallery
        
        if i < n_probe and j < n_gallery and i < 128:
            # Usa shared memory para probe (rápido!)
            px = shared_probe_x[i]
            py = shared_probe_y[i]
            pt = shared_probe_theta[i]
            
            # Gallery da global memory
            gx = gallery_x[j]
            gy = gallery_y[j]
            gt = gallery_theta[j]
            
            # Calcula distância euclidiana
            dx = px - gx
            dy = py - gy
            distance = math.sqrt(dx * dx + dy * dy)
            
            # Calcula diferença angular
            angle_diff = abs(pt - gt)
            if angle_diff > 180.0:
                angle_diff = 360.0 - angle_diff
            
            # Armazena resultados
            distances[idx] = distance
            angles[idx] = angle_diff

@cuda.jit
def compare_compatibility_kernel(probe_distances, probe_angles,
                                gallery_distances, gallery_angles,
                                compatibility_matrix,
                                n_probe, n_gallery,
                                distance_threshold, angle_threshold):
    """
    Kernel para comparar compatibilidade entre probe e gallery
    Baseado no passo 2 do Bozorth3: "Construct Inter-Fingerprint Compatibility Table"
    """
    idx = cuda.grid(1)
    
    if idx < n_probe * n_gallery:
        i = idx // n_gallery
        j = idx % n_gallery
        
        if i < n_probe and j < n_gallery:
            p_dist = probe_distances[i]
            p_angle = probe_angles[i]
            g_dist = gallery_distances[j]
            g_angle = gallery_angles[j]
            
            # Verifica compatibilidade
            dist_diff = abs(p_dist - g_dist)
            angle_diff = abs(p_angle - g_angle)
            
            if dist_diff < distance_threshold and angle_diff < angle_threshold:
                compatibility_matrix[idx] = 1
            else:
                compatibility_matrix[idx] = 0

# ============================================================================
# CLASSE PRINCIPAL
# ============================================================================

class Bozorth3GPUEngine:
    """
    Engine GPU do Bozorth3 seguindo a abordagem do artigo
    
    Etapas (conforme artigo):
    1. Construct Intra-Fingerprint Minutiae Comparison Tables
    2. Construct Inter-Fingerprint Compatibility Table
    3. Traverse and calculate final score
    """
    
    def __init__(self,
                 distance_threshold: float = 15.0,
                 angle_threshold: float = 20.0,
                 threads_per_block: int = 256):
        """
        Args:
            distance_threshold: Threshold para compatibilidade de distância
            angle_threshold: Threshold para compatibilidade de ângulo (graus)
            threads_per_block: Threads por block CUDA
        """
        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA não disponível")
        
        self.distance_threshold = distance_threshold
        self.angle_threshold = angle_threshold
        self.threads_per_block = threads_per_block
        
        print(f"🚀 Bozorth3 GPU Engine inicializado")
        print(f"   Distance threshold: {distance_threshold}")
        print(f"   Angle threshold: {angle_threshold}°")
        print(f"   Threads per block: {threads_per_block}")
    
    def parse_xyt_file(self, xyt_path: str) -> MinutiaeData:
        """Parse arquivo .xyt"""
        data = np.loadtxt(xyt_path, dtype=float)
        
        if data.size == 0:
            raise ValueError(f"Arquivo vazio: {xyt_path}")
        
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        return MinutiaeData(
            x=data[:, 0].astype(np.float32),
            y=data[:, 1].astype(np.float32),
            theta=data[:, 2].astype(np.float32),
            quality=data[:, 3].astype(np.float32) if data.shape[1] > 3 
                   else np.ones(len(data), dtype=np.float32)
        )
    
    def construct_intra_fingerprint_table(self, minutiae: MinutiaeData) -> Tuple[np.ndarray, np.ndarray]:
        """
        ETAPA 1 do Bozorth3 (GPU): Construct Intra-Fingerprint Tables
        Calcula distâncias e ângulos entre todas as minúcias de UMA impressão
        
        Returns:
            (distances, angles) arrays N x N
        """
        n = len(minutiae.x)
        n_comparisons = n * n
        
        # Aloca arrays de saída
        distances = np.zeros(n_comparisons, dtype=np.float32)
        angles = np.zeros(n_comparisons, dtype=np.float32)
        
        # Transfere para GPU
        d_x = cuda.to_device(minutiae.x)
        d_y = cuda.to_device(minutiae.y)
        d_theta = cuda.to_device(minutiae.theta)
        
        d_distances = cuda.to_device(distances)
        d_angles = cuda.to_device(angles)
        
        # Configura grid/blocks
        blocks = (n_comparisons + self.threads_per_block - 1) // self.threads_per_block
        
        # Executa kernel
        calculate_distances_angles_kernel[blocks, self.threads_per_block](
            d_x, d_y, d_theta,
            d_x, d_y, d_theta,  # Mesma impressão para intra
            d_distances, d_angles,
            n, n
        )
        
        # Retorna para CPU
        distances = d_distances.copy_to_host()
        angles = d_angles.copy_to_host()
        
        # Reshape para matriz N x N
        distances = distances.reshape(n, n)
        angles = angles.reshape(n, n)
        
        return distances, angles
    
    def construct_inter_fingerprint_table(self,
                                         probe_minutiae: MinutiaeData,
                                         gallery_minutiae: MinutiaeData) -> Tuple[np.ndarray, np.ndarray]:
        """
        ETAPA 2 do Bozorth3 (GPU): Construct Inter-Fingerprint Tables
        Calcula distâncias e ângulos entre minúcias de DUAS impressões diferentes
        
        Returns:
            (distances, angles) arrays N_probe x N_gallery
        """
        n_probe = len(probe_minutiae.x)
        n_gallery = len(gallery_minutiae.x)
        n_comparisons = n_probe * n_gallery
        
        # Aloca arrays
        distances = np.zeros(n_comparisons, dtype=np.float32)
        angles = np.zeros(n_comparisons, dtype=np.float32)
        
        # Transfere para GPU
        d_probe_x = cuda.to_device(probe_minutiae.x)
        d_probe_y = cuda.to_device(probe_minutiae.y)
        d_probe_theta = cuda.to_device(probe_minutiae.theta)
        
        d_gallery_x = cuda.to_device(gallery_minutiae.x)
        d_gallery_y = cuda.to_device(gallery_minutiae.y)
        d_gallery_theta = cuda.to_device(gallery_minutiae.theta)
        
        d_distances = cuda.to_device(distances)
        d_angles = cuda.to_device(angles)
        
        # Configura grid/blocks
        blocks = (n_comparisons + self.threads_per_block - 1) // self.threads_per_block
        
        # Executa kernel
        calculate_distances_angles_kernel[blocks, self.threads_per_block](
            d_probe_x, d_probe_y, d_probe_theta,
            d_gallery_x, d_gallery_y, d_gallery_theta,
            d_distances, d_angles,
            n_probe, n_gallery
        )
        
        # Retorna para CPU
        distances = d_distances.copy_to_host()
        angles = d_angles.copy_to_host()
        
        # Reshape
        distances = distances.reshape(n_probe, n_gallery)
        angles = angles.reshape(n_probe, n_gallery)
        
        return distances, angles
    
    def calculate_compatibility_score_cpu(self,
                                         probe_distances: np.ndarray,
                                         probe_angles: np.ndarray,
                                         gallery_distances: np.ndarray,
                                         gallery_angles: np.ndarray) -> int:
        """
        ETAPA 3 do Bozorth3 (CPU): Traverse Compatibility Table
        Compara as tabelas e calcula score final
        
        Conforme artigo: "the host will compare the value of the array as a whole"
        """
        n_probe = probe_distances.shape[0]
        n_gallery = gallery_distances.shape[0]
        
        matches = 0
        
        # Compara cada entrada da probe com cada entrada da gallery
        for i in range(n_probe):
            for j in range(i + 1, n_probe):  # Pares únicos
                p_dist = probe_distances[i, j]
                p_angle = probe_angles[i, j]
                
                # Busca correspondência na gallery
                for k in range(n_gallery):
                    for l in range(k + 1, n_gallery):
                        g_dist = gallery_distances[k, l]
                        g_angle = gallery_angles[k, l]
                        
                        # Verifica compatibilidade
                        dist_diff = abs(p_dist - g_dist)
                        angle_diff = abs(p_angle - g_angle)
                        
                        if (dist_diff < self.distance_threshold and 
                            angle_diff < self.angle_threshold):
                            matches += 1
        
        return matches
    
    def one_to_one_match(self, probe_path: str, gallery_path: str) -> MatchResult:
        """
        Matching 1:1 completo (GPU + CPU)
        
        Fluxo conforme artigo:
        1. GPU: Calcula intra-tables (probe e gallery)
        2. GPU: Calcula inter-table
        3. CPU: Compara e calcula score final
        """
        start = time.time()
        
        # Parse minúcias
        probe_minutiae = self.parse_xyt_file(probe_path)
        gallery_minutiae = self.parse_xyt_file(gallery_path)
        
        print(f"\n🔍 Matching 1:1")
        print(f"   Probe: {len(probe_minutiae.x)} minúcias")
        print(f"   Gallery: {len(gallery_minutiae.x)} minúcias")
        
        # ETAPA 1: Intra-fingerprint tables (GPU)
        start_intra = time.time()
        probe_dist, probe_angles = self.construct_intra_fingerprint_table(probe_minutiae)
        gallery_dist, gallery_angles = self.construct_intra_fingerprint_table(gallery_minutiae)
        intra_time = time.time() - start_intra
        
        print(f"   ⏱️  Intra-tables (GPU): {intra_time*1000:.2f}ms")
        
        # ETAPA 2: Inter-fingerprint table (GPU)
        start_inter = time.time()
        inter_dist, inter_angles = self.construct_inter_fingerprint_table(
            probe_minutiae, gallery_minutiae
        )
        inter_time = time.time() - start_inter
        
        print(f"   ⏱️  Inter-table (GPU): {inter_time*1000:.2f}ms")
        
        # ETAPA 3: Comparação final e score (CPU)
        start_score = time.time()
        score = self.calculate_compatibility_score_cpu(
            probe_dist, probe_angles,
            gallery_dist, gallery_angles
        )
        score_time = time.time() - start_score
        
        print(f"   ⏱️  Score calculation (CPU): {score_time*1000:.2f}ms")
        
        total_time = time.time() - start
        
        # Calcula porcentagem
        max_possible = len(probe_minutiae.x) * len(gallery_minutiae.x)
        percentage = (score / max_possible * 100) if max_possible > 0 else 0
        
        print(f"\n✅ Score: {score}")
        print(f"   Percentage: {percentage:.1f}%")
        print(f"   Total time: {total_time*1000:.2f}ms")
        
        return MatchResult(
            probe_id=Path(probe_path).stem,
            candidate_id=Path(gallery_path).stem,
            score=score,
            percentage=percentage,
            time_ms=total_time * 1000
        )
    
    def one_to_many_match(self,
                         probe_path: str,
                         db_path: str,
                         top_k: int = 10,
                         threshold: float = 40.0,
                         percentage_threshold: bool = True) -> List[MatchResult]:
        """
        Matching 1:N (GPU + CPU)
        
        Conforme artigo: "one-to-many method of fingerprint matching"
        """
        db_dir = Path(db_path)
        candidates = list(db_dir.glob('*.xyt'))
        
        if not candidates:
            raise ValueError(f"Nenhum arquivo .xyt encontrado em {db_path}")
        
        print(f"\n🚀 Matching 1:N")
        print(f"   Probe: {Path(probe_path).name}")
        print(f"   Candidatos: {len(candidates)}")
        print(f"   Threshold: {threshold}%")
        
        start_total = time.time()
        
        # Parse probe uma vez
        probe_minutiae = self.parse_xyt_file(probe_path)
        probe_dist, probe_angles = self.construct_intra_fingerprint_table(probe_minutiae)
        
        results = []
        
        # Match contra cada candidato
        for i, candidate_path in enumerate(candidates):
            if (i + 1) % 100 == 0:
                print(f"   Processando: {i+1}/{len(candidates)}...")
            
            try:
                start_match = time.time()
                
                # Parse gallery
                gallery_minutiae = self.parse_xyt_file(str(candidate_path))
                gallery_dist, gallery_angles = self.construct_intra_fingerprint_table(gallery_minutiae)
                
                # Inter-fingerprint (GPU)
                inter_dist, inter_angles = self.construct_inter_fingerprint_table(
                    probe_minutiae, gallery_minutiae
                )
                
                # Score (CPU)
                score = self.calculate_compatibility_score_cpu(
                    probe_dist, probe_angles,
                    gallery_dist, gallery_angles
                )
                
                match_time = time.time() - start_match
                
                # Calcula porcentagem
                max_possible = len(probe_minutiae.x) * len(gallery_minutiae.x)
                percentage = (score / max_possible * 100) if max_possible > 0 else 0
                
                # Filtra por threshold (pode ser score absoluto ou percentage)
                if percentage_threshold:
                    should_include = percentage >= threshold
                else:
                    should_include = score >= threshold
                
                if should_include:
                    results.append(MatchResult(
                        probe_id=Path(probe_path).stem,
                        candidate_id=candidate_path.stem,
                        score=score,
                        percentage=percentage,
                        time_ms=match_time * 1000
                    ))
            
            except Exception as e:
                print(f"   ⚠️  Erro em {candidate_path.name}: {e}")
                continue
        
        total_time = time.time() - start_total
        
        # Ordena por score
        results.sort(key=lambda x: x.score, reverse=True)
        results = results[:top_k]
        
        print(f"\n✅ Matching concluído!")
        print(f"   Total: {total_time:.2f}s")
        print(f"   Velocidade: {len(candidates)/total_time:.1f} comp/seg")
        print(f"   Matches encontrados: {len(results)}")
        
        return results
    
    def benchmark_vs_cpu(self, probe_path: str, gallery_path: str, iterations: int = 10):
        """
        Benchmark GPU vs CPU (simula CPU com numpy)
        Conforme Tabela 1 do artigo
        """
        print(f"\n{'='*60}")
        print(f"BENCHMARK: GPU vs CPU (Bozorth3)")
        print(f"{'='*60}")
        
        # Parse uma vez
        probe_minutiae = self.parse_xyt_file(probe_path)
        gallery_minutiae = self.parse_xyt_file(gallery_path)
        
        # Teste GPU
        print(f"\n🔥 Teste GPU ({iterations} iterações)...")
        gpu_times = []
        for i in range(iterations):
            start = time.time()
            result = self.one_to_one_match(probe_path, gallery_path)
            gpu_times.append(time.time() - start)
        
        avg_gpu = np.mean(gpu_times) * 1000
        
        # Teste CPU (simulado com numpy puro)
        print(f"\n💻 Teste CPU ({iterations} iterações)...")
        cpu_times = []
        for i in range(iterations):
            start = time.time()
            # Simula processamento CPU (sem GPU)
            probe_dist, probe_angles = self._cpu_intra_table(probe_minutiae)
            gallery_dist, gallery_angles = self._cpu_intra_table(gallery_minutiae)
            score = self.calculate_compatibility_score_cpu(
                probe_dist, probe_angles, gallery_dist, gallery_angles
            )
            cpu_times.append(time.time() - start)
        
        avg_cpu = np.mean(cpu_times) * 1000
        
        speedup = avg_cpu / avg_gpu
        
        print(f"\n{'='*60}")
        print(f"RESULTADOS")
        print(f"{'='*60}")
        print(f"GPU time: {avg_gpu:.2f}ms")
        print(f"CPU time: {avg_cpu:.2f}ms")
        print(f"Speedup: {speedup:.2f}x")
        print(f"Score: {result.score}")
        print(f"Percentage: {result.percentage:.1f}%")
    
    def _cpu_intra_table(self, minutiae: MinutiaeData) -> Tuple[np.ndarray, np.ndarray]:
        """Versão CPU do intra-table (para benchmark)"""
        n = len(minutiae.x)
        distances = np.zeros((n, n), dtype=np.float32)
        angles = np.zeros((n, n), dtype=np.float32)
        
        for i in range(n):
            for j in range(n):
                dx = minutiae.x[i] - minutiae.x[j]
                dy = minutiae.y[i] - minutiae.y[j]
                distances[i, j] = np.sqrt(dx*dx + dy*dy)
                
                angle_diff = abs(minutiae.theta[i] - minutiae.theta[j])
                if angle_diff > 180:
                    angle_diff = 360 - angle_diff
                angles[i, j] = angle_diff
        
        return distances, angles


# ============================================================================
# EXEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    if not CUDA_AVAILABLE:
        print("❌ CUDA não disponível. Instale CUDA toolkit e Numba.")
        exit(1)
    
    # Criar engine
    engine = Bozorth3GPUEngine(
        distance_threshold=15.0,
        angle_threshold=20.0,
        threads_per_block=256
    )
    
    # Exemplo 1: Matching 1:1
    print("\n" + "="*60)
    print("TESTE 1: Matching 1:1")
    print("="*60)
    
    probe = "/home/leticia/documentos/sspce/data/db_minutias_nist/0001_L_01_F_L.xyt"
    gallery = "/home/leticia/documentos/sspce/data/db_minutias_nist/0001_L_01_F_L.xyt"
    
    result = engine.one_to_one_match(probe, gallery)
    
    # Exemplo 2: Benchmark
    print("\n" + "="*60)
    print("TESTE 2: Benchmark GPU vs CPU")
    print("="*60)
    
    engine.benchmark_vs_cpu(probe, gallery, iterations=5)
    
    # Exemplo 3: Matching 1:N
    # print("\n" + "="*60)
    # print("TESTE 3: Matching 1:N")
    # print("="*60)
    # 
    # db_path = "/home/leticia/documentos/sspce/data/db_minutias_nist"
    # matches = engine.one_to_many_match(probe, db_path, top_k=5, threshold=40.0)
    # 
    # print("\n🎯 Top Matches:")
    # for i, match in enumerate(matches, 1):
    #     print(f"   #{i}: {match.candidate_id}")
    #     print(f"       Score: {match.score} | Percentage: {match.percentage:.1f}%")