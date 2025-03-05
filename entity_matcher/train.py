import argparse
import logging
import time

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from entity_matcher import EntityMatchingModel

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser(description="Train the Entity Matching Model")
parser.add_argument(
    "--data_path",
    type=str,
    default="data/ds_challenge_alpas.csv",
    help="Path to the dataset",
)
parser.add_argument(
    "--model_path",
    type=str,
    default="models/entity_matcher.joblib",
    help="Path to save the trained model",
)
args = parser.parse_args()

data_path = args.data_path
model_path = args.model_path

# Read data. Use sample to reduce training time as this is a prototype.
logger.info(f"Reading data from {data_path}")
df = pd.read_csv(data_path)
df.columns = ["id", "entity_1", "entity_2", "tag"]
df = df.sample(frac=0.005, random_state=123).reset_index(drop=True)

# Split data
train_data, validation_data = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["tag"]
)

# Initialize, train, and save model
model = EntityMatchingModel(model_path=model_path)

logger.info(f"Training model on {len(train_data)} samples")
start_time = time.monotonic()
train_metrics = model.train(X=train_data[["entity_1", "entity_2"]], y=train_data["tag"])
end_time = time.monotonic()
elapsed_time = end_time - start_time
logger.info(f"Trained model in {elapsed_time:.2f} seconds")

model.save()
logger.info(f"Saved model to {model_path}")

# Evaluate on validation set
y_val_pred, y_val_pred_proba = model.predict(validation_data[["entity_1", "entity_2"]])
y_val = validation_data["tag"]

# Calculate metrics
val_metrics = {
    "accuracy": (y_val_pred == y_val).mean(),
    "precision": precision_score(y_val, y_val_pred),
    "recall": recall_score(y_val, y_val_pred),
    "f1": f1_score(y_val, y_val_pred),
    "roc_auc": roc_auc_score(y_val, y_val_pred_proba),
}

# Print metrics
# fmt: off
logger.info("Training Metrics:")
logger.info(f"Accuracy: {train_metrics['accuracy']:.6f}")
logger.info(f"Positive samples ratio: {train_metrics['positive_samples']:.6f}")
logger.info("Validation Metrics:")
logger.info("Explanation of metrics:")
logger.info(f"Accuracy:  {val_metrics['accuracy']:.6f} Proportion of correct predictions (both positive and negative)")
logger.info(f"Precision: {val_metrics['precision']:.6f} Of the predicted matches, what proportion were actually matches")
logger.info(f"Recall:    {val_metrics['recall']:.6f} Of the actual matches, what proportion were correctly identified")
logger.info(f"F1-Score:  {val_metrics['f1']:.6f} Mean of precision and recall")
logger.info(f"ROC-AUC:   {val_metrics['roc_auc']:.6f} Area under ROC curve, measures model's ability to distinguish between classes")
# fmt: on

# Print example predictions and errors
logger.info("\nExample Predictions:")
predictions = validation_data[["entity_1", "entity_2"]].copy()
predictions["actual"] = y_val
predictions["predicted"] = y_val_pred
predictions["confidence"] = y_val_pred_proba
logger.info(predictions.head(20).to_string())

logger.info("-" * 80)
errors = predictions[predictions.actual != predictions.predicted]
logger.info(
    f"Incorrect Predictions: {len(errors)}/{len(predictions)} -- {round(len(errors)/len(predictions)*100, 2)}%"
)
logger.info(f"\n{errors.to_string()}")
