"""Unit tests for the entity matching API."""

import json

import pytest
from flask import Flask

from entity_matcher.api import app


@pytest.fixture
def client():
    """Fixture to provide a test client for the Flask app."""
    with app.test_client() as client:
        yield client


def test_health_check(client):
    """Test the health check endpoint."""
    response = client.get("/status")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["status"] == "healthy"
    assert data["model"] == "loaded"


@pytest.mark.parametrize(
    "entity_1,entity_2,expected_match,expected_confidence",
    [
        (
            "Apple Inc",
            "Apple Incorporated",
            True,
            0.95,
        ),  # Adjust expected confidence based on your model
        ("Microsoft Corp", "Microsoft Corporation", True, 0.90),
        ("Random Corp", "Totally Different Ltd", False, 0.10),
        ("Tech Solutions GmbH", "Tech Solutions AG", True, 0.85),
    ],
)
def test_predict(client, entity_1, entity_2, expected_match, expected_confidence):
    """Test the predict endpoint."""
    response = client.post(
        "/predict",
        data=json.dumps({"entity_1": entity_1, "entity_2": entity_2}),
        content_type="application/json",
    )
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["is_match"] == expected_match
    assert isinstance(data["confidence"], float)
    assert data["confidence"] >= 0.0 and data["confidence"] <= 1.0
    assert data["entities"]["entity_1"] == entity_1
    assert data["entities"]["entity_2"] == entity_2


@pytest.mark.parametrize(
    "payload,expected_status",
    [
        ({"entity_1": "Company A GmbH"}, 400),  # Missing entity_2
        ({"entity_1": "", "entity_2": ""}, 400),  # Both entities are empty
        ({"entity_1": "Valid Entity", "entity_2": ""}, 400),  # entity_2 is empty
        (
            {"entity_1": "", "entity_2": "Another Valid Entity"},
            400,
        ),  # entity_1 is empty
    ],
)
def test_predict_invalid_payloads(client, payload, expected_status):
    """Test the predict endpoint with various invalid payloads."""
    response = client.post(
        "/predict",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert response.status_code == expected_status
    data = json.loads(response.data)
    assert "error" in data
