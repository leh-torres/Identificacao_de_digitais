// Controla visibilidade do seletor de núcleos
document.addEventListener('DOMContentLoaded', function() {
    const modeSelect = document.getElementById('mode');
    const coresGroup = document.getElementById('cores-group');
    
    // Inicializa visibilidade
    updateCoresVisibility();
    
    // Listener para mudanças
    modeSelect.addEventListener('change', updateCoresVisibility);
    
    function updateCoresVisibility() {
        coresGroup.style.display = modeSelect.value === 'cpu' ? 'block' : 'none';
    }
});

async function executeMatching() {
    const probeFile = document.getElementById('probe-image').files[0];
    const dbPath = 'teste';
    const mode = document.getElementById('mode').value;
    const cores = document.getElementById('cores').value;
    const topK = document.getElementById('top-k').value;
    const threshold = document.getElementById('threshold').value;

    if (!probeFile) {
        alert('❌ Selecione uma imagem probe');
        return;
    }

    if (!dbPath) {
        alert('❌ Informe o diretório da base de dados');
        return;
    }

    // Mostra loading
    document.getElementById('loading').style.display = 'block';
    document.getElementById('results').classList.remove('show');

    try {
        console.log('📤 Fazendo upload da probe...');
        
        // Upload da probe e extração de minúcias
        const formData = new FormData();
        formData.append('file', probeFile);

        const extractResponse = await fetch('/extract', {
            method: 'POST',
            body: formData
        });

        if (!extractResponse.ok) {
            const error = await extractResponse.json();
            throw new Error(error.erro || 'Erro na extração de minúcias');
        }
        
        const extractData = await extractResponse.json();
        console.log('✅ Minúcias extraídas:', extractData.xyt);

        // Converte probe para base64
        const probeBase64 = await fileToBase64(probeFile);

        console.log('🔍 Executando matching...');
        console.log(`   Base: ${dbPath}`);
        console.log(`   Modo: ${mode}`);
        console.log(`   Top K: ${topK}`);
        console.log(`   Threshold: ${threshold}`);

        // Executa matching
        const matchResponse = await fetch('/api/matching', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                probe_xyt: extractData.xyt,
                probe_image: probeBase64,
                db_path: dbPath,
                mode: mode,
                cores: parseInt(cores),
                top_k: parseInt(topK),
                threshold: parseInt(threshold)
            })
        });

        if (!matchResponse.ok) {
            const error = await matchResponse.json();
            throw new Error(error.erro || 'Erro no matching');
        }
        
        const results = await matchResponse.json();
        console.log('✅ Matching concluído:', results);

        displayResults(results, probeBase64);

    } catch (error) {
        console.error('❌ Erro:', error);
        alert('Erro: ' + error.message);
        document.getElementById('loading').style.display = 'none';
    } finally {
        document.getElementById('loading').style.display = 'none';
    }
}

function fileToBase64(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
}

function displayResults(data, probeImage) {
    const resultsDiv = document.getElementById('results');
    
    console.log('📊 Exibindo resultados:', data.matches.length, 'matches');
    
    // Verifica se há matches
    if (data.matches.length === 0) {
        resultsDiv.innerHTML = `
            <div style="text-align: center; padding: 40px; color: #666;">
                <h3>❌ Nenhum match encontrado</h3>
                <p>Tente ajustar o threshold ou verificar a base de dados</p>
            </div>
        `;
        resultsDiv.classList.add('show');
        return;
    }
    
    let html = `
        <div class="stats-bar">
            <div class="stat-item">
                <div class="stat-value">${data.total_comparisons}</div>
                <div class="stat-label">Comparações</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">${data.time.toFixed(2)}s</div>
                <div class="stat-label">Tempo Total</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">${data.speed.toFixed(1)}</div>
                <div class="stat-label">Comp/seg</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">${data.matches.length}</div>
                <div class="stat-label">Matches</div>
            </div>
        </div>

        <h3 style="margin: 20px 0; color: #333;">🎯 Resultados Visuais</h3>
        <div class="images-grid">
    `;

    // Imagem Probe
    html += `
        <div class="image-card probe">
            <img src="${probeImage}" alt="Probe">
            <div class="image-label">🔍 PROBE</div>
            <div style="font-size: 12px; color: #666; margin-top: 5px;">Imagem de busca</div>
        </div>
    `;

    // Matches
    data.matches.forEach((match, idx) => {
        if (match.image_base64) {
            html += `
                <div class="image-card match">
                    <img src="data:image/png;base64,${match.image_base64}" alt="${match.candidate_id}">
                    <div class="image-label">${match.candidate_id}</div>
                    <div class="rank-badge">#${match.rank}</div>
                    <div class="score-badge">Score: ${match.score.toFixed(0)}</div>
                </div>
            `;
        } else {
            // Caso a imagem não seja encontrada
            html += `
                <div class="image-card match" style="opacity: 0.6;">
                    <div style="
                        width: 100%;
                        height: 200px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        background: #f0f0f0;
                        border-radius: 8px;
                        margin-bottom: 10px;
                        color: #999;
                    ">
                        <div>
                            <div style="font-size: 48px;">❌</div>
                            <div style="font-size: 12px; margin-top: 10px;">Imagem não encontrada</div>
                        </div>
                    </div>
                    <div class="image-label">${match.candidate_id}</div>
                    <div class="rank-badge">#${match.rank}</div>
                    <div class="score-badge">Score: ${match.score.toFixed(0)}</div>
                </div>
            `;
        }
    });

    html += '</div>';
    
    resultsDiv.innerHTML = html;
    resultsDiv.classList.add('show');
    
    console.log('✅ Resultados exibidos');
}