"""
Mopdule for entity name matching model operations.
Handles model training, saving, loading, and prediction functionality.
"""

import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


class CosineSimilarityTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to compute cosine similarity between entity pairs.
    Inherits from sklearn's BaseEstimator and TransformerMixin
    for sklearn.pipeline.Pipeline compatibility.
    """

    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))

    def fit(self, X: pd.DataFrame, y=None) -> "CosineSimilarityTransformer":
        """
        Fit the vectorizer on both entity columns.

        Args:
            X: DataFrame with entity columns
            y: Ignored, exists for sklearn compatibility
        """
        all_text = pd.concat([X["entity_1"], X["entity_2"]])
        self.vectorizer.fit(all_text)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform entity pairs to cosine similarity scores."""
        tfidf_1 = self.vectorizer.transform(X["entity_1"])
        tfidf_2 = self.vectorizer.transform(X["entity_2"])
        cos_sim = cosine_similarity(tfidf_1, tfidf_2).diagonal()
        return cos_sim.reshape(-1, 1)


class EntityMatchingModel:
    """
    Main class for entity name matching operations.
    Handles model training, persistence, loading, and predictions.
    """

    def __init__(self, model_path: str | Path = "models/entity_matcher.joblib"):
        """
        Initialize the model with a path for saving/loading.

        Args:
            model_path: Path where model will be saved/loaded from
        """
        self._model_path = Path(model_path)
        self._pipeline: Pipeline = None
        self._is_trained = False
        self._initialize_pipeline()

    def _preprocess_series(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing to all text columns in the dataframe."""
        return X.map(self._preprocess_text)

    def _initialize_pipeline(self) -> None:
        """Initialize the sklearn pipeline with all necessary steps."""
        self._pipeline = Pipeline(
            [
                ("preprocessing", FunctionTransformer(self._preprocess_series)),
                ("cosine_sim", CosineSimilarityTransformer()),
                ("clf", LogisticRegression(solver="saga", n_jobs=-1)),
            ]
        )

    @staticmethod
    def _preprocess_text(text: str) -> str:
        """
        Preprocess text by lowercasing and removing special characters.
        """
        text = text.lower()
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def train(self, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        """
        Train the pipeline on provided data.

        Args:
            X: DataFrame with 'entity_1' and 'entity_2' columns
            y: Series with binary labels

        Returns:
            dictionary with training metrics
        """
        self._pipeline.fit(X, y)
        self._is_trained = True
        y_pred = self._pipeline.predict(X)
        return {"accuracy": (y_pred == y).mean(), "positive_samples": y.mean()}

    def predict(self, entity_pairs: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Validate and make predictions on entity pairs.

        Args:
            entity_pairs: DataFrame with 'entity_1' and 'entity_2' columns

        Returns:
            Tuple of (binary predictions, probability scores)
        """
        self._validate_entity_pairs(entity_pairs)
        predictions = self._pipeline.predict(entity_pairs)
        probabilities = self._pipeline.predict_proba(entity_pairs)[:, 1]
        return predictions, probabilities

    def _validate_entity_pairs(self, entity_pairs: pd.DataFrame) -> None:
        """Validate the entity pairs DataFrame.
        Raises ValueError if the DataFrame is invalid.
        """
        if not all(col in entity_pairs.columns for col in ["entity_1", "entity_2"]):
            raise ValueError(
                "Input DataFrame must contain 'entity_1' and 'entity_2' columns."
            )
        if (
            entity_pairs["entity_1"].str.strip().eq("").any()
            or entity_pairs["entity_2"].str.strip().eq("").any()
        ):
            raise ValueError(
                "Neither 'entity_1' nor 'entity_2' can contain empty strings."
            )
        if not all(isinstance(entity, str) for entity in entity_pairs["entity_1"]):
            raise ValueError("All values in 'entity_1' must be strings.")
        if not all(isinstance(entity, str) for entity in entity_pairs["entity_2"]):
            raise ValueError("All values in 'entity_2' must be strings.")

    def save(self) -> None:
        """Save the trained pipeline to disk."""
        if not self._is_trained:
            raise ValueError("No trained model to save")

        self._model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._pipeline, self._model_path)

    def load(self) -> None:
        """Load a trained pipeline from disk."""
        if not self._model_path.exists():
            raise FileNotFoundError(f"No saved model found at {self._model_path}")

        self._pipeline = joblib.load(self._model_path)
