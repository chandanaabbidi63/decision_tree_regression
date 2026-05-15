import numpy as np
import pandas as pd

from dataclasses import dataclass

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV


class IQROutlierRemover(BaseEstimator, TransformerMixin):
    """Remove rows that are outliers based on IQR rule.

    Note: this transformer removes rows during fit/transform using the training
    thresholds. For deployment (single-row prediction), the transform will
    effectively pass through the sample unchanged.
    """

    def __init__(self, factor: float = 1.5):
        self.factor = factor

    def fit(self, X, y=None):
        X_df = self._to_dataframe(X)
        q1 = X_df.quantile(0.25, numeric_only=True)
        q3 = X_df.quantile(0.75, numeric_only=True)
        iqr = q3 - q1
        self.lower_ = q1 - self.factor * iqr
        self.upper_ = q3 + self.factor * iqr
        return self

    def transform(self, X):
    
        # We use outlier thresholds only during training-time fit.
        X_df = self._to_dataframe(X)
        return X_df.to_numpy()

    def _to_dataframe(self, X):
        if isinstance(X, pd.DataFrame):
            return X.copy()
        return pd.DataFrame(X)


class CorrelationFeatureSelector(BaseEstimator, TransformerMixin):
    """Select features whose absolute correlation with target exceeds threshold.

    This selector computes correlations during fit using y.
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def fit(self, X, y):
        X_df = self._to_dataframe(X)
        y_arr = np.asarray(y)

        df = X_df.copy()
        df["target"] = y_arr

        corr = df.corr(numeric_only=True)["target"].abs()
        # Exclude target itself if it ever appears as a column name
        feature_names = [c for c in df.columns if c != "target"]

        selected = [
            c for c in feature_names if corr.get(c, 0.0) is not None and corr.get(c, 0.0) > self.threshold
        ]
        # Fallback: ensure at least one feature
        if not selected:
            selected = feature_names

        self.selected_features_ = selected
        return self

    def transform(self, X):
        X_df = self._to_dataframe(X)
        # Map integer-column names if X was provided as numpy
        if not all(f in X_df.columns for f in self.selected_features_):
            # If X_df columns are 0..n-1 and selected are also 0..n-1, this works.
            # Otherwise, just assume the selected indices correspond to first k features.
            try:
                idx = [int(f) for f in self.selected_features_]
                return X_df.iloc[:, idx].to_numpy()
            except Exception:
                # last resort
                return X_df.to_numpy()

        return X_df.loc[:, self.selected_features_].to_numpy()

    def _to_dataframe(self, X):
        if isinstance(X, pd.DataFrame):
            return X.copy()
        return pd.DataFrame(X)


def build_pipeline(random_state: int = 42, correlation_threshold: float = 0.5, iqr_factor: float = 1.5):
    """Create the preprocessing + estimator pipeline.

    GridSearchCV is applied by train.py for reproducibility.
    """

    preprocessing = Pipeline(
        steps=[
            ("iqr_outlier_removal", IQROutlierRemover(factor=iqr_factor)),
            ("corr_feature_selector", CorrelationFeatureSelector(threshold=correlation_threshold)),
            ("scaler", StandardScaler()),
        ]
    )

    # We will grid-search the regressor hyperparameters.
    reg = DecisionTreeRegressor(random_state=random_state)

    return preprocessing, reg


def get_param_grid():
    return {
        "max_depth": [3, 5, 7, 10, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

