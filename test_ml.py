import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split

from ml.model import train_model, compute_model_metrics

def test_train_model_uses_adaboost():
    """
    The train_model function should return an AdaBoostClassifier instance.
    """
    # create a tiny dummy dataset
    X = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
    y = np.array([0, 1, 1, 0])
    model = train_model(X, y)
    assert isinstance(model, AdaBoostClassifier), "Expected an AdaBoostClassifier"

def test_compute_model_metrics_perfect():
    """
    compute_model_metrics on perfect predictions should yield 1.0 for precision, recall, and F1.
    """
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0, 1])
    precision, recall, f1 = compute_model_metrics(y_true, y_pred)
    assert precision == 1.0, f"Precision should be 1.0 on perfect predictions, got {precision}"
    assert recall == 1.0,    f"Recall should be 1.0 on perfect predictions, got {recall}"
    assert f1 == 1.0,        f"F1 should be 1.0 on perfect predictions, got {f1}"

def test_train_test_split_dataframe():
    """
    train_test_split should split a DataFrame of length 10 into 8 training and 2 test rows.
    """
    df = pd.DataFrame({"a": range(10), "b": range(10)})
    train, test = train_test_split(df, test_size=0.2, random_state=0)
    # types
    assert isinstance(train, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)
    # sizes
    assert len(train) == 8, f"Expected 8 train rows, got {len(train)}"
    assert len(test) == 2,  f"Expected 2 test rows, got {len(test)}"
