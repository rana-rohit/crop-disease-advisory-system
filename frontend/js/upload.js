/**
 * upload.js — file selection + analyze handler
 * IDs: #imageInput, #previewImage, #analyzeButton
 *
 * BUGFIX (Step 2): Original version relied solely on CSS stacking
 * (#imageInput positioned absolute+opacity:0 under visual elements)
 * to catch clicks. If any sibling element painted on top without
 * pointer-events:none, clicks were silently swallowed and the file
 * picker never opened.
 *
 * Fix is two-layered:
 *   1. CSS: #imageInput given z-index:5 and all decorative children
 *      wrapped in .upload-zone__visual with pointer-events:none.
 *   2. JS (this file): explicit click delegation on the zone container
 *      itself, calling input.click() programmatically. This makes the
 *      zone clickable even if CSS stacking is altered later, and
 *      guarantees keyboard/assistive-tech activation works too.
 */

const imageInput    = document.getElementById('imageInput');
const previewImage  = document.getElementById('previewImage');
const analyzeButton = document.getElementById('analyzeButton');
const uploadZone     = document.getElementById('uploadZone');

// ── Structural click fix: zone always opens picker ──
if (uploadZone && imageInput) {
  uploadZone.addEventListener('click', function (e) {
    // Avoid double-firing if the native input itself was the click target
    if (e.target === imageInput) return;
    imageInput.click();
  });
}

if (imageInput) {
  imageInput.addEventListener('change', function () {
    const file = this.files[0];
    if (!file) return;
    if (previewImage) {
      previewImage.src = URL.createObjectURL(file);
      previewImage.style.display = 'block';
    }
  });
}

if (analyzeButton) {
  analyzeButton.addEventListener('click', async function () {
    const file = imageInput && imageInput.files[0];
    if (!file) {
      const errEl = document.getElementById('errorMessage');
      const errTxt = document.getElementById('errorText');
      if (errEl && errTxt) { errTxt.textContent = 'Please select a leaf image before analyzing.'; errEl.classList.add('visible'); }
      else alert('Please select an image.');
      return;
    }
    analyzeButton.disabled = true;
    try {
      const result = await predictDisease(file);
      localStorage.setItem('predictionResult', JSON.stringify(result));
      const reader = new FileReader();
      reader.onload = function () {
        localStorage.setItem('uploadedImage', reader.result);
        window.location.href = 'result.html';
      };
      reader.readAsDataURL(file);
    } catch (error) {
      console.error('Prediction error:', error);
      const overlay = document.getElementById('loadingOverlay');
      if (overlay) overlay.classList.remove('active');
      analyzeButton.disabled = false;
      analyzeButton.classList.remove('btn--loading');

      const errEl  = document.getElementById('errorMessage');
      const errTxt = document.getElementById('errorText');
      let msg = error.message || 'Prediction failed. Please check the backend and try again.';

      // Friendlier messaging for common network failure (Step 10)
      if (error instanceof TypeError && /fetch/i.test(error.message)) {
        msg = 'Cannot reach the backend server. Please make sure the API is running.';
      }

      if (errEl && errTxt) {
        errTxt.textContent = msg;
        errEl.classList.add('visible');
      } else {
        alert('Prediction failed: ' + msg);
      }
    }
  });
}