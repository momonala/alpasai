"""
Flask API for entity matching service.
Provides endpoints for prediction and health checks.
"""

import logging
from typing import Any, Dict

import pandas as pd
from flask import Flask, request
from werkzeug.exceptions import BadRequest

from entity_matcher.classifier import EntityMatchingModel

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
model = EntityMatchingModel()

try:
    model.load()
    logger.info("Model loaded successfully")
except FileNotFoundError:
    logger.error("No model file found. Please train the model first.")
    raise


@app.route("/status")
def status() -> tuple[Dict[str, str], int]:
    """Health check endpoint."""
    return {"status": "healthy", "model": "loaded"}, 200


@app.route("/predict", methods=["POST"])
def predict() -> tuple[Dict[str, Any], int]:
    """
    Predict endpoint for entity matching.

    This endpoint accepts a JSON payload containing two entities to compare for potential matches.

    Expected request JSON payload:
    {
        "entity_1": "Company A GmbH",
        "entity_2": "Company A AG"
    }

    Response JSON payload:
    Response format:
    - HTTP Status Code
    - Content-Type: application/json
    - Response Body:
        {
            "is_match":     # A boolean indicating if the entities are considered a match.
            "confidence":   # A float representing the confidence level of the prediction.
            "entities": {   # A dict containing the original entities provided in the request.
                "entity_1": "Company A GmbH",
                "entity_2": "Company A AG"
            }
        }

        OR
        {
            "error":  # Error message detailing the issue.
        }
    """
    try:
        data = request.get_json()
        # basic schema validation. Could use Cerberus or Pydantic for more robust validation
        if not data or "entity_1" not in data or "entity_2" not in data:
            raise BadRequest("Missing required fields: entity_1, entity_2")

        entities = pd.DataFrame(
            [{"entity_1": data["entity_1"], "entity_2": data["entity_2"]}]
        )

        prediction, probability = model.predict(entities)

        response = {
            "is_match": bool(prediction[0]),
            "confidence": float(probability[0]),
            "entities": {"entity_1": data["entity_1"], "entity_2": data["entity_2"]},
        }

        logger.info(
            "Prediction made",
            extra={
                "entity_1": data["entity_1"],
                "entity_2": data["entity_2"],
                "prediction": bool(prediction[0]),
                "confidence": float(probability[0]),
            },
        )

        return response, 200

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return {"error": str(e)}, 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
