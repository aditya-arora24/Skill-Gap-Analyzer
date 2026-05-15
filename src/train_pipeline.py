"""
Training pipeline: data loading, feature selection, splitting, and scaling.
Features used: semantic_similarity, num_resume_skills, num_job_skills
Excluded (data leakage): skill_coverage, skill_gap
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── 1. Load dataset ──────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


# ── 2. Select features and target ────────────────────────────────────────────

FEATURES = ["semantic_similarity", "num_resume_skills", "num_job_skills"]
TARGET   = "label"

def select_features(df: pd.DataFrame):
    X = df[FEATURES].copy()
    y = df[TARGET].copy()
    return X, y


# ── 3. Train / test split ─────────────────────────────────────────────────────

def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


# ── 4. Scale features ─────────────────────────────────────────────────────────

def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


# ── 5. Diagnostics ────────────────────────────────────────────────────────────

def print_diagnostics(X: pd.DataFrame, y: pd.Series,
                      X_train, X_test, y_train, y_test):
    print("=" * 50)
    print("DATASET OVERVIEW")
    print("=" * 50)
    print(f"X shape : {X.shape}")
    print(f"y shape : {y.shape}")
    print(f"\nFeatures used: {FEATURES}")

    print("\nClass distribution (full dataset):")
    counts = y.value_counts().sort_index()
    for label, count in counts.items():
        print(f"  Label {label}: {count:>5}  ({count / len(y) * 100:.1f}%)")

    print(f"\nTrain set size : {X_train.shape[0]} samples")
    print(f"Test  set size : {X_test.shape[0]} samples")

    print("\nClass distribution (train):")
    for label, count in y_train.value_counts().sort_index().items():
        print(f"  Label {label}: {count:>5}  ({count / len(y_train) * 100:.1f}%)")

    print("\nClass distribution (test):")
    for label, count in y_test.value_counts().sort_index().items():
        print(f"  Label {label}: {count:>5}  ({count / len(y_test) * 100:.1f}%)")
    print("=" * 50)


# ── Main ──────────────────────────────────────────────────────────────────────

def prepare_data(dataset_path: str):
    df = load_data(dataset_path)
    X, y = select_features(df)

    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    print_diagnostics(X, y, X_train, X_test, y_train, y_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


if __name__ == "__main__":
    DATASET_PATH = "../data/processed/ml_ready_dataset.csv"
    X_train, X_test, y_train, y_test, scaler = prepare_data(DATASET_PATH)
    print("\nData preparation complete. Ready for model training.")
