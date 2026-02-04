/**
 * DL_Bayesian - Frontend JavaScript
 */

const API_BASE = '/api/v1';

/**
 * Check API health and update status indicators
 */
async function checkHealth() {
    try {
        const response = await fetch('/health');
        const data = await response.json();

        // Update health status
        const healthStatus = document.getElementById('health-status');
        if (healthStatus) {
            healthStatus.textContent = data.status === 'healthy' ? 'Operativo' : 'Degradado';
            healthStatus.className = `status-indicator ${data.status}`;
        }

        // Update MLFlow status
        const mlflowStatus = document.getElementById('mlflow-status');
        if (mlflowStatus) {
            const connected = data.checks?.mlflow_connected;
            mlflowStatus.textContent = connected ? 'Conectado' : 'Desconectado';
            mlflowStatus.className = `status-indicator ${connected ? 'healthy' : 'error'}`;
        }

        // Update model status
        const modelStatus = document.getElementById('model-status');
        if (modelStatus) {
            const loaded = data.checks?.model_loaded;
            modelStatus.textContent = loaded ? 'Cargado' : 'No cargado';
            modelStatus.className = `status-indicator ${loaded ? 'healthy' : 'degraded'}`;
        }

        return data;
    } catch (error) {
        console.error('Health check failed:', error);

        // Update all to error state
        const elements = ['health-status', 'mlflow-status', 'model-status'];
        elements.forEach(id => {
            const el = document.getElementById(id);
            if (el) {
                el.textContent = 'Error';
                el.className = 'status-indicator error';
            }
        });

        return null;
    }
}

/**
 * Load registered models for dashboard
 */
async function loadModels() {
    const container = document.getElementById('models-list');
    if (!container) return;

    try {
        const response = await fetch(`${API_BASE}/models`);
        const models = await response.json();

        if (models.length === 0) {
            container.innerHTML = '<li>No hay modelos registrados</li>';
            return;
        }

        container.innerHTML = models.slice(0, 5).map(model => {
            const prodVersion = model.latest_versions?.find(v => v.current_stage === 'Production');
            return `
                <li>
                    <strong>${model.name}</strong>
                    ${prodVersion ? `<span class="version-stage stage-production">v${prodVersion.version}</span>` : ''}
                </li>
            `;
        }).join('');
    } catch (error) {
        container.innerHTML = '<li class="error">Error al cargar modelos</li>';
    }
}

/**
 * Load experiments for dashboard
 */
async function loadExperiments() {
    const container = document.getElementById('experiments-list');
    if (!container) return;

    try {
        const response = await fetch(`${API_BASE}/experiments`);
        const experiments = await response.json();

        if (experiments.length === 0) {
            container.innerHTML = '<li>No hay experimentos</li>';
            return;
        }

        container.innerHTML = experiments.slice(0, 5).map(exp => `
            <li>
                <strong>${exp.name}</strong>
                <span>${exp.runs_count} runs</span>
            </li>
        `).join('');
    } catch (error) {
        container.innerHTML = '<li class="error">Error al cargar experimentos</li>';
    }
}

/**
 * Make a prediction
 */
async function makePrediction(data, resultContainerId = 'quick-result') {
    const container = document.getElementById(resultContainerId);
    if (!container) return;

    try {
        const response = await fetch(`${API_BASE}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Prediction failed');
        }

        const result = await response.json();
        displayPredictionResult(result, resultContainerId);

    } catch (error) {
        container.classList.remove('hidden');
        container.innerHTML = `
            <div class="result-card">
                <div class="result-body">
                    <p class="error">Error: ${error.message}</p>
                </div>
            </div>
        `;
    }
}

/**
 * Display prediction result
 */
function displayPredictionResult(result, containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;

    container.classList.remove('hidden');

    const isSustainable = result.prediction === 1;
    const label = isSustainable ? 'SOSTENIBLE' : 'NO SOSTENIBLE';
    const labelClass = isSustainable ? 'sustainable' : 'not-sustainable';
    const icon = isSustainable ? '&#10004;' : '&#10006;';
    const description = isSustainable
        ? 'La operacion pesquera es sostenible segun el modelo.'
        : 'La operacion pesquera NO es sostenible. Se recomienda revisar los parametros.';

    // For quick predict (simple display)
    if (containerId === 'quick-result') {
        container.innerHTML = `
            <div class="result-card">
                <div class="result-header">
                    <h3>Resultado</h3>
                </div>
                <div class="result-body">
                    <div class="prediction-result">
                        <span class="prediction-label ${labelClass}">${icon} ${label}</span>
                        <span class="prediction-prob">${(result.probability * 100).toFixed(1)}% probabilidad</span>
                    </div>
                    ${result.confidence_interval ? `
                        <div class="uncertainty-info">
                            <span>IC 95%: [${(result.confidence_interval[0] * 100).toFixed(1)}%, ${(result.confidence_interval[1] * 100).toFixed(1)}%]</span>
                        </div>
                    ` : ''}
                </div>
            </div>
        `;
        return;
    }

    // For full predict page
    const predictionIcon = document.getElementById('prediction-icon');
    const predictionLabel = document.getElementById('prediction-label');
    const predictionDesc = document.getElementById('prediction-description');
    const probValue = document.getElementById('probability-value');
    const uncertValue = document.getElementById('uncertainty-value');
    const ciValue = document.getElementById('ci-value');
    const timeValue = document.getElementById('inference-time');
    const modelUsed = document.getElementById('model-used');
    const versionUsed = document.getElementById('version-used');

    if (predictionIcon) predictionIcon.innerHTML = icon;
    if (predictionLabel) {
        predictionLabel.textContent = label;
        predictionLabel.className = `prediction-label-large ${labelClass}`;
    }
    if (predictionDesc) predictionDesc.textContent = description;
    if (probValue) probValue.textContent = `${(result.probability * 100).toFixed(1)}%`;
    if (timeValue) timeValue.textContent = `${result.inference_time_ms.toFixed(1)} ms`;
    if (modelUsed) modelUsed.textContent = result.model_used;
    if (versionUsed) versionUsed.textContent = result.model_version;

    // Uncertainty info (BNN only)
    const uncertMetric = document.getElementById('uncertainty-metric');
    const ciMetric = document.getElementById('ci-metric');

    if (result.uncertainty !== null && result.uncertainty !== undefined) {
        if (uncertValue) uncertValue.textContent = `${(result.uncertainty * 100).toFixed(2)}%`;
        if (uncertMetric) uncertMetric.style.display = 'block';
    } else {
        if (uncertMetric) uncertMetric.style.display = 'none';
    }

    if (result.confidence_interval) {
        if (ciValue) ciValue.textContent = `[${(result.confidence_interval[0] * 100).toFixed(1)}%, ${(result.confidence_interval[1] * 100).toFixed(1)}%]`;
        if (ciMetric) ciMetric.style.display = 'block';
    } else {
        if (ciMetric) ciMetric.style.display = 'none';
    }
}

/**
 * Handle quick prediction form
 */
document.addEventListener('DOMContentLoaded', function() {
    const quickForm = document.getElementById('quick-predict-form');
    if (quickForm) {
        quickForm.addEventListener('submit', async function(e) {
            e.preventDefault();

            const formData = new FormData(this);
            const data = {
                sst_c: parseFloat(formData.get('sst_c')),
                salinity_ppt: parseFloat(formData.get('salinity_ppt')),
                chlorophyll_mg_m3: 2.5,  // Default
                ph: 8.1,  // Default
                fleet_size: parseInt(formData.get('fleet_size')),
                fishing_effort_hours: parseFloat(formData.get('fishing_effort_hours')),
                fuel_consumption_l: 5000,  // Default
                fish_price_usd_ton: 2500,  // Default
                fuel_price_usd_l: 1.2,  // Default
                operating_cost_usd: 15000  // Default
            };

            await makePrediction(data, 'quick-result');
        });
    }
});

/**
 * Format timestamp
 */
function formatTimestamp(timestamp) {
    if (!timestamp) return '-';
    const date = new Date(timestamp);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
}

/**
 * Show loading state
 */
function showLoading(container) {
    if (container) {
        container.innerHTML = '<div class="loading-spinner">Cargando...</div>';
    }
}

/**
 * Show error state
 */
function showError(container, message) {
    if (container) {
        container.innerHTML = `<p class="error">${message}</p>`;
    }
}
