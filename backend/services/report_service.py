"""
Future-ready reporting service.
Placeholder for PDF generation integration.
"""

from backend.utils.exceptions import AppError

def generate_pdf_report(prediction_result: dict, image_bytes: bytes):
    """
    Generate a professional PDF report containing the prediction results,
    advisory information, and image visualizations.
    
    Args:
        prediction_result (dict): The structured result from prediction_service.
        image_bytes (bytes): The original or Grad-CAM overlaid image.
        
    Returns:
        BytesIO: A buffer containing the generated PDF file.
    """
    raise NotImplementedError("PDF report generation is planned for a future release.")
