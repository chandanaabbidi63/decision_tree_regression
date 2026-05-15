import os
import joblib
import numpy as np
import pandas as pd

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

from model_pipeline import build_pipeline, get_param_grid


def main():
    random_state = 42

    diabetes = load_diabetes()
    X = diabetes.data
    y = diabetes.target

    # Keep as DataFrame so selectors can use feature names
    df = pd.DataFrame(X, columns=diabetes.feature_names)

    # Split once for evaluation, matching notebook approach
    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=0.2, random_state=random_state
    )

    preprocessing, reg = build_pipeline(
        random_state=random_state,
        correlation_threshold=0.5,
        iqr_factor=1.5,
    )

    # Fit preprocessing on training data.
    # Important: correlation selector needs y that matches the rows it sees.
    # So we preprocess X_train/X_test using the same fitted preprocessing.
    X_train_proc = preprocessing.fit_transform(X_train, y_train)
    X_test_proc = preprocessing.transform(X_test)


    grid = GridSearchCV(
        estimator=reg,
        param_grid=get_param_grid(),
        cv=5,
        n_jobs=-1,
        scoring="r2",
    )

    grid.fit(X_train_proc, y_train)

    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test_proc)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Best parameters:", grid.best_params_)
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"R2:  {r2:.4f}")

    artifact = {
        "preprocessing": preprocessing,
        "model": best_model,
        "feature_names": diabetes.feature_names,
        "metrics": {"mae": mae, "mse": mse, "r2": r2},
    }

    os.makedirs("models", exist_ok=True)
    out_path = os.path.join("models", "decision_tree_diabetes.joblib")
    joblib.dump(artifact, out_path)
    print("Saved model to:", out_path)


if __name__ == "__main__":
    main()

