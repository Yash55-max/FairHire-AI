from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import shap
from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

SUPPORTED_MODELS = {
    "logistic_regression",
    "random_forest",
    "decision_tree",
    "gradient_boosting",
}


@dataclass
class TrainArtifacts:
    model: Pipeline
    test_frame: pd.DataFrame
    y_true: pd.Series
    y_pred: pd.Series
    metrics: dict[str, Any]
    label_mapping: dict[Any, int] | None = None


# ---------------------------------------------------------------------------
# Column suggestion
# ---------------------------------------------------------------------------

def suggest_target_columns(frame: pd.DataFrame) -> list[str]:
    """Heuristically suggest columns that likely represent the hiring outcome."""
    keywords = ("target", "label", "hired", "selected", "outcome", "decision", "status", "result", "approved")
    candidates: list[str] = []
    for col in frame.columns:
        low = col.lower()
        if any(k in low for k in keywords):
            candidates.append(col)
    # Fallback: binary columns at the end
    if not candidates:
        for col in reversed(frame.columns.tolist()):
            if frame[col].nunique() == 2:
                candidates.append(col)
                if len(candidates) >= 3:
                    break
    if not candidates:
        candidates = list(frame.columns[-3:])
    return candidates[:5]


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_model(model_type: str, numeric_cols: list[str], categorical_cols: list[str]) -> Pipeline:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_pipe, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_pipe, categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    classifier: Any
    if model_type == "logistic_regression":
        classifier = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")
    elif model_type == "decision_tree":
        classifier = DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=42)
    elif model_type == "gradient_boosting":
        classifier = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
    else:  # random_forest (default)
        classifier = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            random_state=42,
            class_weight="balanced_subsample",
            n_jobs=-1,
        )

    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_pipeline(
    frame: pd.DataFrame,
    target_column: str,
    model_type: str,
    test_size: float,
    random_state: int,
) -> TrainArtifacts:
    if model_type not in SUPPORTED_MODELS:
        raise ValueError(f"model_type must be one of {sorted(SUPPORTED_MODELS)}")
    if target_column not in frame.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset")

    dataset = frame.copy()
    y_raw = dataset[target_column]
    X = dataset.drop(columns=[target_column])

    if y_raw.nunique() < 2:
        raise ValueError("Target column must contain at least two distinct classes")

    # Encode target to integers if non-numeric
    label_mapping: dict[Any, int] | None = None
    if y_raw.dtype == object or str(y_raw.dtype).startswith("category"):
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y_raw.astype(str)), index=y_raw.index, name=target_column)
        label_mapping = {cls: int(idx) for idx, cls in enumerate(le.classes_)}
    else:
        y = y_raw.astype(int)

    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    if not numeric_cols and not categorical_cols:
        raise ValueError("No feature columns available after removing the target column")

    model = build_model(model_type=model_type, numeric_cols=numeric_cols, categorical_cols=categorical_cols)

    stratify = y if y.nunique() <= 20 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    model.fit(X_train, y_train)
    y_pred = pd.Series(model.predict(X_test), index=X_test.index, name="prediction")

    avg = "binary" if y.nunique() == 2 else "weighted"
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average=avg, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, average=avg, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, average=avg, zero_division=0)),
    }

    labels = sorted(pd.Series(y).dropna().unique().tolist())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    if cm.size >= 4:
        metrics["confusion_matrix"] = {
            "tp": int(cm[-1, -1]),
            "fn": int(cm[-1, 0]),
            "fp": int(cm[0, -1]),
            "tn": int(cm[0, 0]),
        }
    else:
        metrics["confusion_matrix"] = {"tp": int(cm.sum()), "fn": 0, "fp": 0, "tn": 0}

    test_frame = X_test.copy()
    test_frame[target_column] = y_test

    return TrainArtifacts(
        model=model,
        test_frame=test_frame,
        y_true=pd.Series(y_test, index=X_test.index),
        y_pred=y_pred,
        metrics=metrics,
        label_mapping=label_mapping,
    )


# ---------------------------------------------------------------------------
# Bias
# ---------------------------------------------------------------------------

def compute_bias(
    test_frame: pd.DataFrame,
    y_true: pd.Series,
    y_pred: pd.Series,
    sensitive_column: str,
    positive_label: Any | None = None,
) -> dict[str, Any]:
    if sensitive_column not in test_frame.columns:
        raise ValueError(f"Sensitive column '{sensitive_column}' not found in test data")

    sensitive = test_frame[sensitive_column].astype(str)
    if sensitive.nunique() < 2:
        raise ValueError(f"Sensitive column '{sensitive_column}' needs at least two distinct groups")

    if positive_label is None:
        sorted_labels = sorted(pd.Series(y_true).dropna().unique().tolist(), key=str)
        positive_label = sorted_labels[-1]

    mf = MetricFrame(
        metrics={
            "selection_rate": selection_rate,
            "true_positive_rate": true_positive_rate,
        },
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive,
    )

    sr = mf.by_group["selection_rate"].fillna(0.0)
    tpr = mf.by_group["true_positive_rate"].fillna(0.0)
    dp_diff = float(sr.max() - sr.min())
    eo_diff = float(tpr.max() - tpr.min())
    fairness_index = float(max(0.0, 1.0 - ((dp_diff + eo_diff) / 2.0)))

    return {
        "sensitive_column": sensitive_column,
        "demographic_parity_difference": dp_diff,
        "equal_opportunity_difference": eo_diff,
        "selection_rate_by_group": {str(k): float(v) for k, v in sr.items()},
        "true_positive_rate_by_group": {str(k): float(v) for k, v in tpr.items()},
        "fairness_index": fairness_index,
        "positive_label": positive_label,
    }


# ---------------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------------

def compute_explainability(model: Pipeline, test_frame: pd.DataFrame, sample_size: int = 40) -> dict[str, Any]:
    if sample_size < 1:
        raise ValueError("sample_size must be >= 1")

    sample = test_frame.head(sample_size).copy()
    preprocessor: ColumnTransformer = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    transformed = preprocessor.transform(sample)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    transformed = np.asarray(transformed, dtype=float)

    try:
        feature_names = preprocessor.get_feature_names_out().tolist()
    except AttributeError:
        feature_names = [f"feature_{i}" for i in range(transformed.shape[1])]

    # Choose the right SHAP explainer
    try:
        if isinstance(classifier, (RandomForestClassifier, GradientBoostingClassifier, DecisionTreeClassifier)):
            explainer = shap.TreeExplainer(classifier)
            shap_vals = explainer.shap_values(transformed)
            # For multi-output tree explainers shap_values returns a list
            if isinstance(shap_vals, list):
                values = np.array(shap_vals[-1])  # last class (positive)
            else:
                values = np.array(shap_vals)
            if values.ndim == 3:
                values = values[..., -1]
        else:
            explainer_gen = shap.Explainer(classifier, transformed, feature_names=feature_names)
            sv = explainer_gen(transformed)
            values = sv.values
            if values.ndim == 3:
                values = values[..., -1]
    except Exception:  # noqa: BLE001 – fallback to linear explainer
        explainer_lin = shap.LinearExplainer(classifier, transformed)
        sv = explainer_lin(transformed)
        values = sv.values
        if values.ndim == 3:
            values = values[..., -1]

    global_importance = np.mean(np.abs(values), axis=0)
    top_idx = np.argsort(global_importance)[::-1][:10]

    top_global_features = [
        {
            "feature": feature_names[i],
            "mean_abs_shap": float(global_importance[i]),
            "importance": float(global_importance[i]),  # alias for frontend charts
        }
        for i in top_idx
    ]

    local_idx = np.argsort(np.abs(values[0]))[::-1][:8]
    local_explanation = [
        {
            "feature": feature_names[i],
            "shap_value": float(values[0][i]),
            "direction": "positive" if values[0][i] >= 0 else "negative",
        }
        for i in local_idx
    ]

    return {
        "sample_size": int(sample.shape[0]),
        "top_global_features": top_global_features,
        "local_explanation": local_explanation,
    }
