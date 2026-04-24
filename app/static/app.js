const recordsBody = document.getElementById("records-body");
const benchmarkBody = document.getElementById("benchmark-body");
const probabilityBars = document.getElementById("probability-bars");

const modelModeLabels = {
    "trained-checkpoint": "Trained Local Checkpoint",
    "prototype-baseline": "Fallback Similarity Model",
};

function percent(value) {
    return `${Number(value).toFixed(1)}%`;
}

function riskClass(riskBand) {
    return `risk-${riskBand.toLowerCase()}`;
}

function createBars(containerId, items, formatter = (value) => value) {
    const container = document.getElementById(containerId);
    container.innerHTML = "";
    for (const item of items) {
        const row = document.createElement("div");
        row.className = "bar-row";
        row.innerHTML = `
            <div class="bar-meta">
                <span>${item.label}</span>
                <strong>${formatter(item.value)}</strong>
            </div>
            <div class="bar-track">
                <div class="bar-fill" style="width:${Math.max(4, (item.value / Math.max(...items.map((entry) => entry.value), 1)) * 100)}%; background:${item.color || "#c4572e"}"></div>
            </div>
        `;
        container.appendChild(row);
    }
}

function renderPrediction(payload) {
    document.getElementById("result-label").textContent = `${payload.predicted_class} - ${payload.predicted_label}`;
    const riskNode = document.getElementById("result-risk");
    riskNode.textContent = payload.risk_band;
    riskNode.className = riskClass(payload.risk_band);
    document.getElementById("result-percentage").textContent = percent(payload.cancer_percentage);
    document.getElementById("result-confidence").textContent = percent(payload.confidence_score);
    document.getElementById("result-source").textContent = modelModeLabels[payload.model_source] || payload.model_source;
    document.getElementById("result-card").classList.remove("hidden");
    probabilityBars.innerHTML = "";
    for (const item of payload.class_probabilities) {
        const row = document.createElement("div");
        row.className = "bar-row";
        row.innerHTML = `
            <div class="bar-meta">
                <span>${item.class_code}</span>
                <strong>${percent(item.probability)}</strong>
            </div>
            <div class="bar-track">
                <div class="bar-fill" style="width:${Math.max(item.probability, 3)}%; background:${item.color}"></div>
            </div>
        `;
        probabilityBars.appendChild(row);
    }
}

function renderRecords(records) {
    recordsBody.innerHTML = "";
    if (!records.length) {
        recordsBody.innerHTML = `<tr><td colspan="7">No prediction records yet.</td></tr>`;
        return;
    }

    for (const record of records) {
        const row = document.createElement("tr");
        row.innerHTML = `
            <td>${record.filename}</td>
            <td>${record.predicted_class}</td>
            <td>${percent(record.cancer_percentage)}</td>
            <td class="${riskClass(record.risk_band)}">${record.risk_band}</td>
            <td>${percent(record.confidence_score)}</td>
            <td>${modelModeLabels[record.model_source] || record.model_source}</td>
            <td>${new Date(record.created_at).toLocaleString()}</td>
        `;
        recordsBody.appendChild(row);
    }
}

function renderBenchmark(report) {
    benchmarkBody.innerHTML = "";
    for (const item of report) {
        const row = document.createElement("tr");
        row.innerHTML = `
            <td>${item.class_code}</td>
            <td>${item.precision.toFixed(2)}</td>
            <td>${item.recall.toFixed(2)}</td>
            <td>${item.f1_score.toFixed(2)}</td>
        `;
        benchmarkBody.appendChild(row);
    }
}

async function loadDashboard() {
    const response = await fetch("/api/dashboard");
    const payload = await response.json();

    document.getElementById("model-mode").textContent = modelModeLabels[payload.model_status.mode] || payload.model_status.mode;
    document.getElementById("model-note").textContent = payload.model_status.checkpoint_available
        ? "A validated local checkpoint is active for inference."
        : payload.model_status.checkpoint_found
            ? `A trained checkpoint is available, but the system is currently using the more stable fallback model for live predictions.`
            : "No validated checkpoint was found, so the fallback similarity model is active.";

    document.getElementById("dataset-total").textContent = payload.dataset.total_images.toLocaleString();
    document.getElementById("clinical-total").textContent = payload.clinical.total_patients.toLocaleString();
    document.getElementById("benchmark-accuracy").textContent = percent(payload.benchmark.test_accuracy * 100);
    document.getElementById("prediction-total").textContent = payload.prediction_summary.total_predictions.toLocaleString();

    document.getElementById("train-accuracy").textContent = percent(payload.benchmark.train_accuracy * 100);
    document.getElementById("valid-accuracy").textContent = percent(payload.benchmark.valid_accuracy * 100);
    document.getElementById("test-accuracy").textContent = percent(payload.benchmark.test_accuracy * 100);
    document.getElementById("test-loss").textContent = payload.benchmark.test_loss.toFixed(3);

    document.getElementById("mean-age").textContent = payload.clinical.mean_age;
    const maleCount = payload.clinical.sex_distribution.find((item) => item.label === "Male")?.value || 0;
    const femaleCount = payload.clinical.sex_distribution.find((item) => item.label === "Female")?.value || 0;
    document.getElementById("male-count").textContent = maleCount;
    document.getElementById("female-count").textContent = femaleCount;

    createBars("dataset-bars", payload.dataset.class_distribution, (value) => value.toLocaleString());
    createBars("stage-bars", payload.clinical.stage_distribution, (value) => value);
    createBars("histology-bars", payload.clinical.top_histologies.map((item) => ({ ...item, color: "#c4572e" })), (value) => value);
    renderBenchmark(payload.benchmark.class_report);
    renderRecords(payload.recent_predictions);

    const biomarkerBody = document.getElementById("biomarker-body");
    biomarkerBody.innerHTML = "";
    for (const target of payload.biomarkers.top_targets) {
        const row = document.createElement("tr");
        row.innerHTML = `
            <td>${target.target_symbol}</td>
            <td>${target.predicted_sum}</td>
            <td>${target.all_sum}</td>
        `;
        biomarkerBody.appendChild(row);
    }
}

async function runPrediction(event) {
    event.preventDefault();
    const fileInput = document.getElementById("file-input");
    if (!fileInput.files.length) {
        return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    const response = await fetch("/api/predict", {
        method: "POST",
        body: formData,
    });
    const payload = await response.json();

    if (!response.ok) {
        alert(payload.detail || "Prediction failed.");
        return;
    }

    renderPrediction(payload);
    const latestPredictions = await fetch("/api/predictions");
    const predictionPayload = await latestPredictions.json();
    renderRecords(predictionPayload.records);
    document.getElementById("prediction-total").textContent = predictionPayload.summary.total_predictions.toLocaleString();
}

document.getElementById("upload-form").addEventListener("submit", runPrediction);
document.getElementById("refresh-btn").addEventListener("click", loadDashboard);
window.addEventListener("DOMContentLoaded", loadDashboard);
