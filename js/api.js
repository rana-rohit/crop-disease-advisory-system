/**
 * api.js — Backend communication
 * Single function: predictDisease(imageFile) -> Promise<result>
 * API contract unchanged from original.
 */
const API_BASE_URL =
  window.CROPSENSE_API_BASE_URL || 'https://crop-disease-advisory-system.onrender.com/api';

async function predictDisease(imageFile) {
  const formData = new FormData();
  formData.append('image', imageFile);

  const response = await fetch(`${API_BASE_URL}/predict`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errorBody = await response.json().catch(() => ({}));
    throw new Error(errorBody.error || `Server error ${response.status}`);
  }

  return await response.json();
}

async function generateGradCam(imageFile, classIndex) {
  const formData = new FormData();
  formData.append('image', imageFile);
  if (classIndex !== undefined && classIndex !== null) {
    formData.append('class_index', classIndex);
  }

  const response = await fetch(`${API_BASE_URL}/xai/gradcam`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errorBody = await response.json().catch(() => ({}));
    throw new Error(errorBody.error || `Server error ${response.status}`);
  }

  return await response.json();
}