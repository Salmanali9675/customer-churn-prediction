# Customer Churn Prediction System

An end-to-end Machine Learning portfolio project that predicts whether a telecommunications customer is likely to churn (cancel service) based on contract, billing, tenure, and subscribed services.

## Project Structure
```
customer-churn-prediction/
?
??? data/              # Raw Telco Customer Churn dataset
??? notebooks/         # Exploratory Data Analysis (EDA) notebooks
??? src/               # Training script and data processing pipeline
?   ??? train.py
??? model/             # Serialized trained model and scaler artifacts
?   ??? churn_model.pkl
??? app.py             # Interactive Streamlit Web Application
??? requirements.txt   # Required Python dependencies
??? README.md          # Project documentation
```

## Machine Learning Results
- **Algorithm:** Logistic Regression (Optimized baseline)
- **Accuracy:** 80.38%
- **ROC-AUC Score:** 0.8357
- **Evaluation Highlights:** Optimized for Churn Recall (57%) and F1-Score (0.61) on an imbalanced dataset (73% Stay vs 27% Churn).

## How to Run Locally

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Train the model:
   ```bash
   python src/train.py
   ```

3. Launch the Streamlit Web Application:
   ```bash
   streamlit run app.py
   ```
