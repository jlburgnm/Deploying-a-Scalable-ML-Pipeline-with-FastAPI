from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import compute_model_metrics, inference, train_model


DATA_PATH = Path(__file__).resolve().parent / "data" / "census.csv"
LABEL = "salary"
CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


@pytest.fixture(scope="module")
def census_data():
    """Load and validate the project's actual census dataset."""
    assert DATA_PATH.exists(), f"Missing dataset: {DATA_PATH}"

    data = pd.read_csv(DATA_PATH)
    data = data.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    expected_columns = {
        "age",
        "workclass",
        "fnlgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "salary",
    }

    assert expected_columns.issubset(data.columns)
    assert not data.empty
    return data


@pytest.fixture(scope="module")
def prepared_data(census_data):
    """Split and preprocess real census rows."""
    train_data, test_data = train_test_split(
        census_data,
        test_size=0.20,
        random_state=42,
        stratify=census_data[LABEL],
    )

    X_train, y_train, encoder, lb = process_data(
        X=train_data,
        categorical_features=CAT_FEATURES,
        label=LABEL,
        training=True,
    )

    X_test, y_test, _, _ = process_data(
        X=test_data,
        categorical_features=CAT_FEATURES,
        label=LABEL,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    return X_train, y_train, X_test, y_test


def test_census_data_has_expected_rows_and_columns(census_data):
    """The repository contains a non-empty
    census dataset with salary labels."""
    assert len(census_data) > 0
    assert LABEL in census_data.columns
    assert set(census_data[LABEL].dropna().unique()) == {"<=50K", ">50K"}


def test_process_data_returns_compatible_arrays(prepared_data):
    """Training and test data have compatible processed feature shapes."""
    X_train, y_train, X_test, y_test = prepared_data

    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert X_train.shape[1] == X_test.shape[1]
    assert X_train.shape[1] > 0
    assert np.isfinite(X_train).all()
    assert np.isfinite(X_test).all()


def test_deterministic_compute_model_metrics():
    """Verify metric calculations match expected mathematical outcomes deterministically."""
    y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 1, 0, 0, 1, 0, 0, 0])

    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
    assert precision == pytest.approx(0.75, abs=1e-6)
    assert recall == pytest.approx(0.60, abs=1e-6)
    assert fbeta == pytest.approx(0.6666666666666666, abs=1e-6)


def test_model_inference_and_metrics_on_census_data(prepared_data):
    """The trained model predicts valid labels and metrics on held-out data."""
    X_train, y_train, X_test, y_test = prepared_data

    model = train_model(X_train, y_train)
    predictions = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, predictions)

    assert len(predictions) == len(y_test)
    assert set(np.unique(predictions)).issubset({0, 1})
    assert 0.0 <= precision <= 1.0
    assert 0.0 <= recall <= 1.0
    assert 0.0 <= fbeta <= 1.0
