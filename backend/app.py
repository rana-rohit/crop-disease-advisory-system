"""
Main application entry point.
"""

from flask import Flask, jsonify
from flask_cors import CORS

from backend.config.settings import get_settings
from backend.utils.logger import get_logger
from backend.utils.exceptions import AppError
from backend.model.model_loader import init_model
from backend.routes.prediction_routes import prediction_bp
from backend.routes.health_routes import health_bp

def create_app():
    """Application factory pattern."""
    settings = get_settings()
    logger = get_logger(__name__)
    
    app = Flask(__name__)
    cors_origins = settings.CORS_ORIGINS if settings.CORS_ORIGINS != "*" else "*"
    CORS(app, resources={r"/api/*": {"origins": cors_origins}})
    
    # Configure Flask limits
    app.config["MAX_CONTENT_LENGTH"] = settings.MAX_CONTENT_LENGTH
    
    # Register error handlers
    @app.errorhandler(AppError)
    def handle_app_error(error):
        logger.warning(f"AppError: {str(error)}")
        return jsonify({"error": str(error)}), error.status_code
        
    @app.errorhandler(413)
    def request_entity_too_large(error):
        logger.warning("Upload rejected: Request entity too large.")
        return jsonify({"error": f"File exceeds maximum allowed size of {settings.MAX_UPLOAD_SIZE_MB}MB."}), 413

    @app.errorhandler(404)
    def handle_not_found(error):
        return jsonify({"error": "Endpoint not found."}), 404
        
    @app.errorhandler(Exception)
    def handle_generic_error(error):
        logger.error(f"Unhandled Exception: {str(error)}", exc_info=True)
        return jsonify({"error": "An unexpected internal server error occurred."}), 500

    # Initialize the model at startup
    with app.app_context():
        init_model()

    # Register blueprints
    from backend.routes.xai_routes import xai_bp
    app.register_blueprint(health_bp, url_prefix="/api")
    app.register_blueprint(prediction_bp, url_prefix="/api")
    app.register_blueprint(xai_bp, url_prefix="/api")
    
    logger.info("Application factory created successfully.")
    return app

# Gunicorn expects an application instance
app = create_app()

if __name__ == "__main__":
    settings = get_settings()
    app.run(
        debug=settings.FLASK_DEBUG,
        host=settings.HOST,
        port=settings.PORT,
    )
