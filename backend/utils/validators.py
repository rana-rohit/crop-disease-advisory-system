"""
Validation utilities for incoming requests.
"""

from werkzeug.datastructures import FileStorage
from PIL import Image, ImageFile
from backend.config.settings import get_settings
from backend.utils.exceptions import ImageValidationError

ImageFile.LOAD_TRUNCATED_IMAGES = False

def validate_image_upload(request) -> FileStorage:
    """
    Validate the image upload from a Flask request.
    Raises ImageValidationError if validation fails.
    Returns the validated FileStorage object.
    """
    settings = get_settings()
    Image.MAX_IMAGE_PIXELS = settings.MAX_IMAGE_PIXELS
    
    if "image" not in request.files:
        raise ImageValidationError("No image file provided in the request.")
        
    file = request.files["image"]
    
    if file.filename == "":
        raise ImageValidationError("No file selected for uploading.")
        
    if "." not in file.filename:
        raise ImageValidationError("File has no extension.")
        
    ext = file.filename.rsplit(".", 1)[1].lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise ImageValidationError(
            f"Invalid file type. Allowed extensions are: {', '.join(settings.ALLOWED_EXTENSIONS)}"
        )
        
    try:
        # verify() checks image structure; load() forces decoding enough to catch bombs/truncation.
        img = Image.open(file.stream)
        img.verify()
        file.stream.seek(0)
        img = Image.open(file.stream)
        img.load()
        if img.width * img.height > settings.MAX_IMAGE_PIXELS:
            raise ImageValidationError("Image dimensions exceed the maximum allowed pixel count.")
        file.stream.seek(0)
    except ImageValidationError:
        raise
    except Exception:
        raise ImageValidationError("The uploaded file is not a valid image or is corrupted.")
        
    return file
