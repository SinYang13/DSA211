"""
Shared Data Loading & Splitting Utilities for DSA211 Project
- Single source of truth for the chronological holdout split (month-aligned, 12.5% held out: 2012-10-01 to 2012-12-31).
- Notebook 00 is the only notebook that reads the raw dataset and writes the split files.
- Notebooks 01-05 read the training set only; the locked test set is read once in Notebook 06.
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / 'data'
RAW_PATH = DATA_DIR / 'hour.csv'
TRAIN_PATH = DATA_DIR / 'train_clean.csv'
TEST_PATH = DATA_DIR / 'test_locked.csv'
BASE_TRAIN_PATH = DATA_DIR / 'train_base.csv'

SPLIT_DATE = pd.Timestamp('2012-10-01')

def _read(path):
    df = pd.read_csv(path)
    df['dteday'] = pd.to_datetime(df['dteday'])
    return df

def load_raw():
    """Loads the full raw hourly dataset. Only Notebook 00 should call this."""
    return _read(RAW_PATH)

def chronological_split(df, split_date=SPLIT_DATE):
    """Splits rows before split_date into train and rows on/after it into test."""
    train_mask = df['dteday'] < split_date
    train_df = df[train_mask].copy().reset_index(drop=True)
    test_df = df[~train_mask].copy().reset_index(drop=True)
    return train_df, test_df

def save_split(train_df, test_df):
    """Writes the train and locked test sets to data/."""
    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)
    return TRAIN_PATH, TEST_PATH

def load_train():
    """Loads the training set written by Notebook 00."""
    return _read(TRAIN_PATH)

def load_train_base():
    """Loads the base processed training dataset written by Notebook 02."""
    return _read(BASE_TRAIN_PATH)

def load_test():
    """Loads the locked test set. Only Notebook 06 should call this, once."""
    return _read(TEST_PATH)

def add_readable_units(df):
    """Adds un-normalized weather columns (deg C, %, km/h) for interpretation."""
    data = df.copy()
    data['temp_c'] = data['temp'] * 41.0
    data['atemp_c'] = data['atemp'] * 50.0
    data['hum_pct'] = data['hum'] * 100.0
    data['windspeed_kmh'] = data['windspeed'] * 67.0
    return data
