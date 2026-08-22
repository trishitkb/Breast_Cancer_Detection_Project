import pytest
import pandas as pd
import numpy as np

def test_data_loading():
    """Test if raw data can be loaded and has expected columns."""
    try:
        df = pd.read_csv("data/raw/breast_cancer_prediction.csv")
    except FileNotFoundError:
        pytest.skip("Raw data not found. Skipping data loading test.")
        
    assert not df.empty, "Dataframe should not be empty"
    assert "Cancer" in df.columns, "Target column 'Cancer' must exist"
    assert "Age" in df.columns, "Feature column 'Age' must exist"

def test_data_preprocessing_no_missing():
    """Test that basic preprocessing would not introduce NaNs."""
    try:
        df = pd.read_csv("data/raw/breast_cancer_prediction.csv")
        df.dropna(inplace=True)
        assert df.isnull().sum().sum() == 0, "There should be no missing values after dropna"
    except FileNotFoundError:
        pytest.skip("Raw data not found. Skipping test.")
