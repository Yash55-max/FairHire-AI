from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
try:
    from xgboost import XGBClassifier
except Exception:  # noqa: BLE001
    XGBClassifier = None
try:
    import shap
except Exception:  # noqa: BLE001
    shap = None

try:
    from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate
except Exception:  # noqa: BLE001
    MetricFrame = None
    selection_rate = None
    true_positive_rate = None
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class TrainArtifacts:
    model: Pipeline
    test_frame: pd.DataFrame
    y_true: pd.Series
    y_pred: pd.Series
    metrics: dict[str, Any]


SUPPORTED_MODELS = {
    "logistic_regression",
    "random_forest",
    "gradient_boosting",
    "decision_tree",
    "xgboost",
}

LEAKY_NAME_KEYWORDS = {
    "decision",
    "selected",
    "selection",
    "hired",
    "hire",
    "outcome",
    "label",
    "target",
    "probability",
    "score_final",
    "final_score",
    "final",
    "result",
    "approved",
    "rejected",
}

ID_NAME_KEYWORDS = {"id", "uuid", "guid", "candidate_id", "record_id", "application_id"}


def suggest_target_columns(frame: pd.DataFrame) -> list[str]:
    candidates: list[str] = []
    keywords = ("target", "label", "hired", "selected", "outcome", "decision")
    for col in frame.columns:
        lowered = col.lower()
        if any(k in lowered for k in keywords):
            candidates.append(col)
    if not candidates:
        candidates = list(frame.columns[-3:])
    return candidates[:5]


def _with_scaler(model_type: str) -> bool:
    return model_type == "logistic_regression"


def build_model(model_type: str, numeric_cols: list[str], categorical_cols: list[str]) -> Pipeline:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler() if _with_scaler(model_type) else "passthrough"),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ]
    )

    if model_type == "logistic_regression":
        classifier = LogisticRegression(max_iter=4000, class_weight="balanced")
    elif model_type == "decision_tree":
        classifier = DecisionTreeClassifier(random_state=42, class_weight="balanced")
    elif model_type == "gradient_boosting":
        classifier = GradientBoostingClassifier(random_state=42)
    elif model_type == "xgboost":
        if XGBClassifier is None:
            raise ValueError("model_type='xgboost' requires the xgboost package. Install it and retry.")
        classifier = XGBClassifier(
            n_estimators=320,
            max_depth=4,
            learning_rate=0.06,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=1,
        )
    else:
        classifier = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            random_state=42,
            class_weight="balanced_subsample",
        )

    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def _parameter_grid(model_type: str) -> dict[str, list[Any]] | None:
    if model_type == "logistic_regression":
        return {
            "classifier__C": [0.25, 1.0, 4.0],
        }
    if model_type == "random_forest":
        return {
            "classifier__n_estimators": [250, 400],
            "classifier__max_depth": [10, 16, None],
            "classifier__min_samples_leaf": [1, 2],
        }
    if model_type == "decision_tree":
        return {
            "classifier__max_depth": [5, 10, None],
            "classifier__min_samples_leaf": [1, 2, 4],
        }
    if model_type == "gradient_boosting":
        return {
            "classifier__n_estimators": [100, 200],
            "classifier__learning_rate": [0.03, 0.08, 0.15],
            "classifier__max_depth": [2, 3],
        }
    if model_type == "xgboost":
        return {
            "classifier__n_estimators": [220, 320],
            "classifier__max_depth": [3, 4, 5],
            "classifier__learning_rate": [0.04, 0.06, 0.1],
            "classifier__subsample": [0.85, 1.0],
        }
    return None


def _add_logistic_interactions(X: pd.DataFrame, y: pd.Series, max_features: int = 2) -> tuple[pd.DataFrame, list[str]]:
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    if len(numeric_cols) < 2:
        return X, []

    target_num = pd.to_numeric(y, errors="coerce").fillna(0.0)
    ranked = sorted(
        numeric_cols,
        key=lambda c: abs(float(pd.to_numeric(X[c], errors="coerce").fillna(0.0).corr(target_num) or 0.0)),
        reverse=True,
    )
    top = ranked[:max_features]
    if len(top) < 2:
        return X, []

    interaction_name = f"int_{top[0]}_x_{top[1]}"
    if interaction_name in X.columns:
        return X, []

    X_enhanced = X.copy()
    X_enhanced[interaction_name] = pd.to_numeric(X[top[0]], errors="coerce").fillna(0.0) * pd.to_numeric(X[top[1]], errors="coerce").fillna(0.0)
    return X_enhanced, [interaction_name]


def _fit_with_tuning(model: Pipeline, model_type: str, X_train: pd.DataFrame, y_train: pd.Series, random_state: int) -> tuple[Pipeline, dict[str, Any]]:
    grid = _parameter_grid(model_type)
    if not grid:
        model.fit(X_train, y_train)
        return model, {}

    class_counts = pd.Series(y_train).value_counts(dropna=False)
    if class_counts.empty:
        model.fit(X_train, y_train)
        return model, {}

    cv_splits = min(5, int(class_counts.min()))
    if cv_splits < 2:
        model.fit(X_train, y_train)
        return model, {}

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    search = GridSearchCV(
        estimator=model,
        param_grid=grid,
        scoring="f1_weighted",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )

    try:
        search.fit(X_train, y_train)
        std_index = int(search.best_index_) if hasattr(search, "best_index_") else 0
        std_scores = search.cv_results_.get("std_test_score", []) if hasattr(search, "cv_results_") else []
        std_best = float(std_scores[std_index]) if len(std_scores) > std_index else 0.0
        return search.best_estimator_, {
            "cv_best_score": float(search.best_score_),
            "cv_score_std": std_best,
            "cv_best_params": search.best_params_,
            "cv_folds": cv_splits,
        }
    except Exception:
        # Keep the train endpoint resilient for small or noisy datasets.
        model.fit(X_train, y_train)
        return model, {}


def _is_identifier_column(name: str, series: pd.Series) -> bool:
    lowered = name.lower()
    n_rows = max(int(series.shape[0]), 1)
    unique_ratio = float(series.nunique(dropna=False) / n_rows)
    has_keyword = any(token in lowered for token in ID_NAME_KEYWORDS)
    return has_keyword or unique_ratio > 0.98


def _is_exact_target_proxy(series: pd.Series, y: pd.Series) -> bool:
    left = series.fillna("<nan>").astype(str).reset_index(drop=True)
    right = y.fillna("<nan>").astype(str).reset_index(drop=True)
    if left.shape[0] != right.shape[0]:
        return False
    return float((left == right).mean()) >= 0.995


def _sanitize_features(X: pd.DataFrame, y: pd.Series, target_column: str) -> tuple[pd.DataFrame, list[str], list[str]]:
    dropped: list[str] = []
    notes: list[str] = []
    y_label = y.fillna("<nan>").astype(str)
    cleaned = X.copy()
    target_lower = target_column.lower()

    for col in list(cleaned.columns):
        series = cleaned[col]
        lowered = col.lower()

        suspicious_name = any(token in lowered for token in LEAKY_NAME_KEYWORDS) and target_lower in lowered
        if suspicious_name:
            dropped.append(col)
            notes.append(f"Dropped '{col}' due to target-like naming pattern.")
            cleaned = cleaned.drop(columns=[col])
            continue

        if _is_identifier_column(col, series):
            dropped.append(col)
            notes.append(f"Dropped '{col}' because it behaves like an identifier/high-cardinality key.")
            cleaned = cleaned.drop(columns=[col])
            continue

        if _is_exact_target_proxy(series, y):
            dropped.append(col)
            notes.append(f"Dropped '{col}' because it is an almost exact proxy of the target column.")
            cleaned = cleaned.drop(columns=[col])
            continue

        if series.nunique(dropna=False) <= 30:
            probe = pd.DataFrame({"x": series.fillna("<nan>").astype(str), "y": y_label})
            group_majority = probe.groupby("x")["y"].agg(lambda grp: grp.value_counts(normalize=True).iloc[0])
            group_weight = probe.groupby("x").size() / max(len(probe), 1)
            proxy_score = float((group_majority * group_weight).sum())
            if proxy_score >= 0.995:
                dropped.append(col)
                notes.append(f"Dropped '{col}' because it almost deterministically maps to target labels (proxy score {proxy_score:.3f}).")
                cleaned = cleaned.drop(columns=[col])

    if cleaned.shape[1] == 0:
        raise ValueError("All feature columns were removed due to leakage/identifier checks. Please upload richer non-proxy features.")

    return cleaned, dropped, notes


def _build_validation_notes(metrics: dict[str, Any], row_count: int) -> list[str]:
    notes: list[str] = []
    accuracy = float(metrics.get("accuracy", 0.0))
    f1_weighted = float(metrics.get("f1_score", 0.0))
    cv_best = float(metrics.get("cv_best_score", 0.0)) if "cv_best_score" in metrics else None

    if row_count < 500:
        notes.append(f"Dataset has {row_count} rows; metrics can be optimistic on small samples. Consider >500 rows for stable estimates.")

    if accuracy >= 0.99 and f1_weighted >= 0.99:
        notes.append("Near-perfect holdout metrics detected. Verify there is no leakage and that labels are not rule-derived from one feature.")

    if cv_best is not None and cv_best >= 0.99 and accuracy >= 0.99:
        notes.append("Cross-validation is also near-perfect, suggesting the dataset may be highly separable or synthetic.")

    if cv_best is not None and abs(accuracy - cv_best) >= 0.08:
        notes.append("Large gap between holdout and CV scores detected. This can indicate split variance or overfitting.")

    if cv_best is not None and abs(accuracy - cv_best) <= 0.03:
        notes.append("Stable cross-validation performance indicates the model generalizes well with minor expected variance.")

    return notes


def train_pipeline(
    frame: pd.DataFrame,
    target_column: str,
    model_type: str,
    test_size: float,
    random_state: int,
) -> TrainArtifacts:
    if target_column not in frame.columns:
        raise ValueError(f"Target column '{target_column}' not found")

    dataset = frame.copy()
    y = dataset[target_column]
    X = dataset.drop(columns=[target_column])

    if y.nunique() < 2:
        raise ValueError("Target column must contain at least two classes")

    X, dropped_columns, leakage_notes = _sanitize_features(X, y, target_column)

    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    if not numeric_cols and not categorical_cols:
        raise ValueError("No feature columns available after removing target")

    selected_model_type = model_type if model_type in SUPPORTED_MODELS else "random_forest"
    interaction_features: list[str] = []
    if selected_model_type == "logistic_regression":
        X, interaction_features = _add_logistic_interactions(X, y)
        numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = [c for c in X.columns if c not in numeric_cols]

    model = build_model(model_type=selected_model_type, numeric_cols=numeric_cols, categorical_cols=categorical_cols)

    stratify = y if y.nunique() <= 20 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    model, tuning_meta = _fit_with_tuning(
        model=model,
        model_type=selected_model_type,
        X_train=X_train,
        y_train=y_train,
        random_state=random_state,
    )
    y_pred = pd.Series(model.predict(X_test), index=X_test.index)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
    }
    metrics.update(tuning_meta)
    metrics["leakage_dropped_columns"] = dropped_columns
    metrics["validation_notes"] = leakage_notes + _build_validation_notes(metrics, row_count=int(dataset.shape[0]))
    metrics["diagnostics"] = list(metrics["validation_notes"])
    metrics["feature_count"] = int(X.shape[1])
    metrics["train_rows"] = int(X_train.shape[0])
    metrics["test_rows"] = int(X_test.shape[0])
    metrics["interaction_features"] = interaction_features

    labels = list(pd.Series(y).dropna().unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    if cm.size >= 4:
        metrics["confusion_matrix"] = {
            "tp": int(cm[-1, -1]),
            "fn": int(cm[-1, 0]),
            "fp": int(cm[0, -1]),
            "tn": int(cm[0, 0]),
        }
    else:
        metrics["confusion_matrix"] = {
            "tp": int(cm.sum()),
            "fn": 0,
            "fp": 0,
            "tn": 0,
        }

    test_frame = X_test.copy()
    test_frame[target_column] = y_test

    return TrainArtifacts(
        model=model,
        test_frame=test_frame,
        y_true=pd.Series(y_test, index=X_test.index),
        y_pred=y_pred,
        metrics=metrics,
    )


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
        raise ValueError(f"Sensitive column '{sensitive_column}' needs at least two groups")

    inferred_positive = positive_label
    if inferred_positive is None:
        sorted_labels = sorted(pd.Series(y_true).dropna().unique().tolist(), key=lambda x: str(x))
        inferred_positive = sorted_labels[-1]

    used_fairlearn = False
    if MetricFrame is not None and selection_rate is not None and true_positive_rate is not None:
        try:
            metrics = MetricFrame(
                metrics={
                    "selection_rate": selection_rate,
                    "true_positive_rate": true_positive_rate,
                },
                y_true=y_true,
                y_pred=y_pred,
                sensitive_features=sensitive,
            )
            sr = metrics.by_group["selection_rate"].fillna(0.0)
            tpr = metrics.by_group["true_positive_rate"].fillna(0.0)
            used_fairlearn = True
        except Exception:
            # Fall back to manual aggregation if metric helpers do not support
            # the target label shape or class structure for the current run.
            used_fairlearn = False

    if not used_fairlearn:
        aligned = pd.DataFrame(
            {
                "sensitive": sensitive,
                "y_true": pd.Series(y_true).astype(str),
                "y_pred": pd.Series(y_pred).astype(str),
            }
        )
        positive = str(inferred_positive)
        sr_map: dict[str, float] = {}
        tpr_map: dict[str, float] = {}
        for group, group_df in aligned.groupby("sensitive", dropna=False):
            total = max(1, int(group_df.shape[0]))
            selected = int((group_df["y_pred"] == positive).sum())
            positives = int((group_df["y_true"] == positive).sum())
            true_positives = int(((group_df["y_true"] == positive) & (group_df["y_pred"] == positive)).sum())
            sr_map[str(group)] = float(selected / total)
            tpr_map[str(group)] = float(true_positives / positives) if positives > 0 else 0.0
        sr = pd.Series(sr_map, dtype=float)
        tpr = pd.Series(tpr_map, dtype=float)
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
        "positive_label": inferred_positive,
    }


def compute_explainability(model: Pipeline, test_frame: pd.DataFrame, sample_size: int = 40) -> dict[str, Any]:
    if sample_size < 1:
        raise ValueError("sample_size must be >= 1")

    sample = test_frame.head(sample_size).copy()
    preprocessor: ColumnTransformer = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    feature_frame = sample

    transformed = preprocessor.transform(sample)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    feature_names = preprocessor.get_feature_names_out().tolist()

    if shap is not None:
        # SHAP can return class-wise values for classification models; collapse to one contribution vector.
        explainer = shap.Explainer(classifier, transformed, feature_names=feature_names)
        shap_values = explainer(transformed)
        values = shap_values.values

        if values.ndim == 3:
            values = values[..., -1]

        global_importance = np.mean(np.abs(values), axis=0)
        top_indices = np.argsort(global_importance)[::-1][:10]

        top_global_features = [
            {
                "feature": feature_names[idx],
                "mean_abs_shap": float(global_importance[idx]),
            }
            for idx in top_indices
        ]

        local_indices = np.argsort(np.abs(values[0]))[::-1][:8]
        local_explanation = [
            {
                "feature": feature_names[idx],
                "shap_value": float(values[0][idx]),
                "direction": "positive" if values[0][idx] >= 0 else "negative",
            }
            for idx in local_indices
        ]
    else:
        # Fallback when SHAP is unavailable: use model-native feature weights.
        if hasattr(classifier, "feature_importances_"):
            global_importance = np.abs(np.asarray(classifier.feature_importances_).ravel())
        elif hasattr(classifier, "coef_"):
            coef = np.asarray(classifier.coef_)
            global_importance = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef).ravel()
        else:
            global_importance = np.abs(np.asarray(transformed).mean(axis=0)).ravel()

        if global_importance.shape[0] != len(feature_names):
            global_importance = np.resize(global_importance, len(feature_names))

        top_indices = np.argsort(global_importance)[::-1][:10]
        top_global_features = [
            {
                "feature": feature_names[idx],
                "mean_abs_shap": float(global_importance[idx]),
            }
            for idx in top_indices
        ]

        row_vector = np.asarray(transformed[0]).ravel()
        contribution = np.abs(row_vector * global_importance)
        local_indices = np.argsort(contribution)[::-1][:8]
        local_explanation = [
            {
                "feature": feature_names[idx],
                "shap_value": float(contribution[idx]),
                "direction": "positive" if row_vector[idx] >= 0 else "negative",
            }
            for idx in local_indices
        ]

    return {
        "sample_size": int(sample.shape[0]),
        "top_global_features": top_global_features,
        "local_explanation": local_explanation,
    }
