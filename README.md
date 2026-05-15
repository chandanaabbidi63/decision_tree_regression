# Diabetes Decision Tree Regression (Streamlit)

This project trains a **DecisionTreeRegressor** on sklearn’s **Diabetes** dataset and deploys it with **Streamlit**.

## Files
- `train.py` - trains the model and saves `models/decision_tree_diabetes.joblib`
- `app.py` - Streamlit UI for predictions
- `model_pipeline.py` - preprocessing + model-building utilities

## Setup
```bash
pip install -r requirements.txt
```

## Train the model (one time)
```bash
python train.py
```
This creates:
- `models/decision_tree_diabetes.joblib`

## Run Streamlit
```bash
streamlit run app.py
```

## Usage
Enter the 10 diabetes feature values and click **Predict**.

## Notes
- Preprocessing in training includes:
  - IQR-based outlier handling (training-time only)
  - correlation-based feature selection (threshold=0.5)
  - StandardScaler

