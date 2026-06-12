"""
Flask application entry point.
"""

from flask import Flask

from backend.routes.prediction_routes import (
    prediction_bp,
)

app = Flask(__name__)

app.register_blueprint(prediction_bp)


@app.route("/")
def home():

    return {
        "message": "Crop Disease Advisory System API"
    }


if __name__ == "__main__":

    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000,
    )