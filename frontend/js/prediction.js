/**
 * prediction.js — Populates result.html with data from localStorage.
 * Preserves all original IDs and behavior. No API contract changes.
 *
 * IDs set by this file (unchanged from original):
 *   #uploadedImage, #diseaseName, #confidence
 *   #symptoms, #treatment, #prevention
 *   #topPredictions, #modelVersion, #predictionTime
 *   #xaiPanel, #gradcamOverlay, #gradcamMeta
 *
 * Additive only: reads result.xai.metadata.activation_stats if present
 * (optional backend field) to populate the new XAI stats card —
 * falls back gracefully to '—' if not provided, never throws.
 */

function formatDiseaseName(name) {
  return name.replaceAll('___', ' — ').replaceAll('_', ' ');
}

// Set uploaded image
const uploadedImage = localStorage.getItem('uploadedImage');
if (uploadedImage) {
  const imgEl = document.getElementById('uploadedImage');
  if (imgEl) imgEl.src = uploadedImage;
}

// Guard: redirect if no result
const result = JSON.parse(localStorage.getItem('predictionResult') || 'null');
if (!result) {
  window.location.href = 'index.html';
}

// Disease name
const diseaseName = document.getElementById('diseaseName');
if (diseaseName && result) {
  diseaseName.textContent = formatDiseaseName(result.prediction.disease);
}

// Confidence (format: "94.3%")
const confidenceEl = document.getElementById('confidence');
if (confidenceEl && result) {
  confidenceEl.textContent = result.prediction.confidence + '%';
}

// Advisory
const symptomsEl = document.getElementById('symptoms');
if (symptomsEl && result) symptomsEl.textContent = result.advisory.symptoms;

const treatmentEl = document.getElementById('treatment');
if (treatmentEl && result) treatmentEl.textContent = result.advisory.treatment;

const preventionEl = document.getElementById('prevention');
if (preventionEl && result) preventionEl.textContent = result.advisory.prevention;

// Top predictions list
const listEl = document.getElementById('topPredictions');
if (listEl && result && result.top_predictions) {
  result.top_predictions.forEach(pred => {
    const item = document.createElement('li');
    item.textContent = `${formatDiseaseName(pred.disease)} (${pred.confidence}%)`;
    listEl.appendChild(item);
  });
}

// Metadata
const modelVersionEl = document.getElementById('modelVersion');
if (modelVersionEl && result) {
  modelVersionEl.textContent = result.metadata?.model_version || 'Unavailable';
}

const predTimeEl = document.getElementById('predictionTime');
if (predTimeEl && result) {
  predTimeEl.textContent = result.metadata?.generated_at || 'Unavailable';
}

// XAI — Grad-CAM
const xaiPanel       = document.getElementById('xaiPanel');
const gradcamOverlay = document.getElementById('gradcamOverlay');
const gradcamMeta    = document.getElementById('gradcamMeta');

if (result && result.xai && result.xai.visualizations && result.xai.visualizations.overlay_base64) {
  if (gradcamOverlay) {
    gradcamOverlay.src = `data:image/png;base64,${result.xai.visualizations.overlay_base64}`;
  }
  if (gradcamMeta) {
    const xaiMeta = result.xai.metadata || {};
    gradcamMeta.textContent = `${xaiMeta.method || 'Grad-CAM'} · ${xaiMeta.target_layer || 'Last Conv Layer'}`;
  }
  if (xaiPanel) xaiPanel.style.display = '';

  // Additive: activation statistics, only if backend supplies them
  const stats = (result.xai.metadata && result.xai.metadata.activation_stats) || null;
  if (stats) {
    const meanEl = document.getElementById('xaiMeanActivation');
    const maxEl  = document.getElementById('xaiMaxActivation');
    if (meanEl && stats.mean_activation !== undefined) meanEl.textContent = Number(stats.mean_activation).toFixed(3);
    if (maxEl  && stats.max_activation  !== undefined) maxEl.textContent  = Number(stats.max_activation).toFixed(3);
  }
} else if (xaiPanel) {
  xaiPanel.style.display = 'none';
}