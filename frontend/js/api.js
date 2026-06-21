/**
 * api.js — Backend communication
 * Single function: predictDisease(imageFile) -> Promise<result>
 * API contract unchanged from original.
 */
const API_BASE_URL =
  window.CROPSENSE_API_BASE_URL || 'http://127.0.0.1:5000/api';

async function predictDisease(imageFile) {
  const formData = new FormData();
  formData.append('image', imageFile);

  const response = await fetch(`${API_BASE_URL}/predict?include_xai=true`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errorBody = await response.json().catch(() => ({}));
    throw new Error(errorBody.error || `Server error ${response.status}`);
  }

  return await response.json();
}