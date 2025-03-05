# Entity Name Resolution System

A system for resolving entity names across different data sources, built with modern Python practices and MLOps principles.

## Core Features

- **ML Pipeline**: Character-level TF-IDF vectorization with cosine similarity for company name matching
- **RESTful API**: Flask-based service with health checks and structured logging
- **Docker Support**: Multi-stage builds with layer caching optimization
- **Unit Testing**: Parameterized unit tests for both model and API endpoints
- **Dependency Management**: Poetry for dependency management

## Technical Stack

- Python 3.12
- scikit-learn for ML pipeline
- Flask for API service
- Poetry for dependency management
- pytest for testing
- Docker for containerization

## Architecture

```
├── explore.ipynb
├── Dockerfile
├── README.md
├── data
│   └── ds_challenge_alpas.csv
├── entity_matcher
│   ├── __init__.py
│   ├── api.py
│   ├── classifier.py
│   └── train.py
├── explore.ipynb
├── models
│   └── entity_matcher.joblib
├── poetry.lock
├── pyproject.toml
└── tests
    ├── test_api.py
    └── test_model.py
```

### Current Workflow

```
Training Pipeline:
┌──────────────┐   ┌──────────────┐    ┌──────────────┐    ┌────────────┐
│ Training Data│──▶│ Train Script │───▶│ Fit Pipeline │───▶│ Save Model │
└──────────────    └──────────────┘    └──────────────┘    └────────────┘
                          ▼                                       │
                   ┌────────────┐                                 │
                   │ Validation │                                 │
                   │ Metrics    │                                 │
                   └────────────┘                                 ▼
Serving Pipeline:                                           ┌───────────┐
                   ┌──────────────┐     ┌──────────────┐    │ Flask API │
                   │    Client    │────▶│ HTTP Request │───▶┤(Container)│
                   └──────────────┘     └──────────────┘    └───────────┘
                          ▲                                      ▼
                   ┌──────────────┐                        ┌────────────┐
                   │   Response   │◀───────────────────────│ Prediction │
                   └──────────────┘                        └────────────┘
```

### Data exploration 
- `explore.ipynb` includes some initial exploration of the data
- Look for initial patterns and get an idea of the dataset
- Propose a model pipeline based on the initial exploration (see below)
- Train a first version of the model and evaluate it
- Take those learnings into the more scalaable code in the `entity_matcher` module

### Model Pipeline
- Custom transformer (`CosineSimilarityTransformer`) implementing scikit-learn's transformer interface in `entity_matcher/classifier.py`
- Character-level n-gram (2-4) TF-IDF vectorization
- Cosine similarity computation for entity pairs
- Logistic regression for final binary classification
- Saves model to `models/entity_matcher.joblib`

note: This model setup is was chosen because it's best to start simple when building something from scratch. This setup is fast, has explainable confidence intervals, and performs well enough. However, it was clear from the unit testing that the model cannot handle renames of companies (ex. Google -> Alphabet) and abbreviations (ex. IBM -> International Business Machines). A more sophisticated model like LLMs would be able to handle this, at the cost of latency and money.

### API Service
- RESTful endpoints for predictions in `entity_matcher/api.py`
- Health checks for monitoring
- Structured logging for observability
- Input validation and error handling
- Reads trained model from `models/entity_matcher.joblib`

### Tests
- Unit tests for executing on trained model and some edge cases in `tests/*.py`
- Unit tests on model training pipeline
- Unit tests for API endpoints

### Docker
- Multi-stage builds with layer caching optimization
- Accepts `models/entity_matcher.joblib` for usage as a volume mount, allowing for smaller dockerfile and flexibility in deployment and local testing

## Usage

### Training

To train the model, copy the .csv data file to the data/ directory and run the training script.

```bash
python -m entity_matcher.train --data_path data/ds_challenge_alpas.csv
```

### Running the API
```bash
docker build -t entity-matcher .
docker run -p 8000:8000 -v $(pwd)/models:/app/models entity-matcher
```

### Endpoints
Health Check

```bash
curl http://localhost:8000/status
```

Prediction
```bash
curl -X POST http://localhost:8000/predict \
-H "Content-Type: application/json" \
-d '{"entity_1": "Tech Corp GmbH", "entity_2": "Tech Corp AG"}'
```

## API Improvements for Production
in no particular order...
- Add a queue (ex. RabbitMQ) for incoming requests to handle multiple requests concurrently
- Add a load balancer (Nginx + Gunicorn) in front of the API to distribute requests across multiple instances
- Add a distributed cache (Redis) in front of the API to cache predictions for frequently requested entities
- Add authentication for API endpoints
- Automated and continuous testing and deployment (CI/CD)
- Add a logging/monitoring system to monitor the API and the model
    - save low confidence predictions to a database for manual review
    - track model performance (percision, recall, F1 score). Tie this to model version.
    - track request characteristics (latency, cache hits, errors)
    - setup alerts for performance degradation, latency, error, system overload
- Dont host model locally. Use a cloud based model hosted on AWS, GCP, etc.
- Depending on usecase, have a batch processing endpoint for bulk entity matching
- Experiment with more sophisticated models (ex. transformers)

## MLOps Workflow

Question: Given that the operational goal of this model pipeline is to flag any named
entity pair that is dissimilar for further quality assurance (QA), how would you
design a system to train, test, and deploy the model to meet the expected
goal?

Answer: 

- The system should prioritize high recall (catch as many mismatches as possible) over high precision, since false positives will be handled by QA.
- The system should save to a database all predictions and their confidence scores.
- The low confidence scores should be flagged, correctly annotated, and allocated for preferential retraining
- We could also add an endpoint for flagging false positives and false negatives
    - save those to the database for further review and retraining of the model
- Retraining could happen once per week, with a bias of data towards the flagged errors at the decision boundary (0.4–0.8 confidence range), since those pairs are most ambiguous.
- Retraining would only need to happen on the Logistic Regression model, not the vectorization or cosine similarity steps, since those are deterministic and invariant to the data.
- We could also retrain using different hyperparameters, perhaps using a grid search to find the best hyperparameters for the model.
