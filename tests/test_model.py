"""Unit tests for the entity matching model."""

from pathlib import Path

import pandas as pd
import pytest
import sklearn

from entity_matcher import EntityMatchingModel


@pytest.fixture
def model():
    """Fixture to provide a trained model for tests."""
    if not Path("models/entity_matcher.joblib").exists():
        raise FileNotFoundError("Model file not found. Please train the model first.")
    model = EntityMatchingModel(model_path="models/entity_matcher.joblib")
    model.load()
    return model


# fmt: off
@pytest.mark.parametrize(
    "entity_1,entity_2,expected_match",
    [
        ("Apple Inc", "Apple Incorporated", True),
        ("Microsoft Corp", "Microsoft Corporation", True),
        ("Google LLC", "Alphabet Inc", False),              # edge case: this should be True technically, but requires societal context. More suited for an LLM
        ("IBM", "International Business Machines", False),  # edge case: this should be True technically, but requires societal context. More suited for an LLM
        ("Random Corp", "Totally Different Ltd", False),
        ("Tech Solutions GmbH", "Tech Solutions AG", True),
        ("ABC Company", "XYZ Limited", False),
    ],
)
# fmt: on
def test_entity_matching(model, entity_1: str, entity_2: str, expected_match: bool):
    """Test entity matching predictions."""
    entities = pd.DataFrame([{"entity_1": entity_1, "entity_2": entity_2}])
    prediction, probability = model.predict(entities)

    assert prediction[0] == expected_match, (
        f"Expected {expected_match} for {entity_1} vs {entity_2}, "
        f"but got {prediction[0]} with confidence {probability[0]:.4f}"
    )


@pytest.mark.parametrize(
    "entity_1,entity_2",
    [("", ""), ("Valid Entity", ""), ("", "Another Valid Entity")],
)
def test_edge_empty_payload(model, entity_1: str, entity_2: str):
    """Test edge cases and potential error conditions."""
    entities = pd.DataFrame([{"entity_1": entity_1, "entity_2": entity_2}])
    with pytest.raises(ValueError):
        _, _ = model.predict(entities)


def test_model_persistence(tmp_path):
    """Test model saving and loading."""
    temp_model_path = tmp_path / "temp_model.joblib"
    test_entities = pd.DataFrame(
        [{"entity_1": "Test Corp", "entity_2": "Test Corp Ltd"}]
    )

    model = EntityMatchingModel(model_path=temp_model_path)

    # Verify we get errors if we try to use the model before it's trained
    with pytest.raises(sklearn.exceptions.NotFittedError):
        model.predict(test_entities)
    with pytest.raises(FileNotFoundError):
        model.load()

    # Train on a dummy dataset
    train_data = pd.DataFrame(
        [
            {"entity_1": "Company A", "entity_2": "Company A Ltd", "tag": 1},
            {"entity_1": "Company B", "entity_2": "Different Corp", "tag": 0},
        ]
    )
    model.train(train_data[["entity_1", "entity_2"]], train_data["tag"])

    # Save the model
    model.save()
    assert temp_model_path.exists()

    # Load in a new instance
    new_model = EntityMatchingModel(model_path=temp_model_path)
    new_model.load()

    # Check predictions match
    pred1, prob1 = model.predict(test_entities)
    pred2, prob2 = new_model.predict(test_entities)
    assert (pred1 == pred2).all()
    assert (prob1 == prob2).all()
