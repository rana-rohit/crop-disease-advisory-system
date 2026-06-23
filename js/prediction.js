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

// Always show the panel now, but with the Generate button
if (xaiPanel) xaiPanel.style.display = '';

const generateXaiBtn = document.getElementById('generateXaiBtn');
const xaiLoading = document.getElementById('xaiLoading');
const xaiErrorMsg = document.getElementById('xaiErrorMsg');
const xaiGenerateContainer = document.getElementById('xaiGenerateContainer');
const xaiResultsContainer = document.getElementById('xaiResultsContainer');

if (generateXaiBtn) {
  generateXaiBtn.addEventListener('click', async () => {
    generateXaiBtn.style.display = 'none';
    xaiLoading.style.display = 'block';
    xaiErrorMsg.style.display = 'none';

    try {
      const dataUrl = localStorage.getItem('uploadedImage');
      if (!dataUrl) throw new Error("No image found");
      const fetchRes = await fetch(dataUrl);
      const blob = await fetchRes.blob();
      const file = new File([blob], "image.jpg", { type: blob.type });
      
      const xaiData = await generateGradCam(file);
      
      if (xaiData && xaiData.xai && xaiData.xai.visualizations && xaiData.xai.visualizations.overlay_base64) {
        if (gradcamOverlay) {
          gradcamOverlay.src = `data:image/png;base64,${xaiData.xai.visualizations.overlay_base64}`;
        }
        if (gradcamMeta) {
          const xaiMeta = xaiData.xai.metadata || {};
          gradcamMeta.textContent = `${xaiMeta.method || 'Grad-CAM'} · ${xaiMeta.target_layer || 'Last Conv Layer'}`;
        }
        
        // Additive: activation statistics
        const stats = (xaiData.xai.metadata && xaiData.xai.metadata.activation_stats) || null;
        if (stats) {
          const meanEl = document.getElementById('xaiMeanActivation');
          const maxEl  = document.getElementById('xaiMaxActivation');
          if (meanEl && stats.mean_activation !== undefined) meanEl.textContent = Number(stats.mean_activation).toFixed(3);
          if (maxEl  && stats.max_activation  !== undefined) maxEl.textContent  = Number(stats.max_activation).toFixed(3);
        }
        
        // Show results, hide generate container
        if (xaiGenerateContainer) xaiGenerateContainer.style.display = 'none';
        if (xaiResultsContainer) xaiResultsContainer.style.display = 'block';
        
        // Ensure fallback images are updated
        const gradcamOverlayFallback = document.getElementById('gradcamOverlayFallback');
        if (gradcamOverlayFallback && gradcamOverlay) gradcamOverlayFallback.src = gradcamOverlay.src;
      } else {
        throw new Error("Invalid XAI data returned");
      }
    } catch (error) {
      console.error('XAI generation failed:', error);
      if (xaiLoading) xaiLoading.style.display = 'none';
      if (generateXaiBtn) generateXaiBtn.style.display = 'inline-block';
      if (xaiErrorMsg) {
        xaiErrorMsg.textContent = "Explanation generation failed. Prediction results remain valid.";
        xaiErrorMsg.style.display = 'block';
      }
    }
  });
}