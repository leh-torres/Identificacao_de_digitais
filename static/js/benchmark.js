async function executeBenchmark() {
    const probeFile = document.getElementById('bench-probe').files[0];
    const dbPath = document.getElementById('bench-db').value;

    if (!probeFile || !dbPath) {
        alert('Preencha todos os campos');
        return;
    }

    document.getElementById('bench-loading').style.display = 'block';
    document.getElementById('bench-results').innerHTML = '';

    try {
        // Upload probe
        const formData = new FormData();
        formData.append('file', probeFile);

        const extractResponse = await fetch('/extract', {
            method: 'POST',
            body: formData
        });

        const extractData = await extractResponse.json();

        // Executa benchmark
        const response = await fetch('/api/benchmark', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                probe_xyt: extractData.xyt,
                db_path: dbPath
            })
        });

        const data = await response.json();
        displayBenchmarkResults(data);

    } catch (error) {
        alert('Erro: ' + error.message);
    } finally {
        document.getElementById('bench-loading').style.display = 'none';
    }
}

function displayBenchmarkResults(data) {
    const resultsDiv = document.getElementById('bench-results');
    let html = '<h2 style="margin-bottom: 20px;">Resultados</h2>';

    for (const [mode, result] of Object.entries(data.results)) {
        html += `
            <div class="result-card">
                <h3>${mode.replace('_', ' ').toUpperCase()}</h3>
                <div class="metric-row">
                    <span class="metric-label">Tempo:</span>
                    <span class="metric-value">${result.time.toFixed(2)}s</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Velocidade:</span>
                    <span class="metric-value">${result.speed.toFixed(1)} comp/seg</span>
                </div>
        `;

        if (result.speedup) {
            html += `
                <div class="metric-row">
                    <span class="metric-label">Speedup:</span>
                    <span class="metric-value">${result.speedup.toFixed(2)}x</span>
                </div>
            `;
        }

        if (result.efficiency) {
            html += `
                <div class="metric-row">
                    <span class="metric-label">Eficiência:</span>
                    <span class="metric-value">${result.efficiency.toFixed(1)}%</span>
                </div>
            `;
        }

        html += '</div>';
    }

    resultsDiv.innerHTML = html;
}