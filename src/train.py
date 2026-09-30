import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

def train():
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'telco_churn.csv')
    df = pd.read_csv(data_path)
    print('Dataset loaded. Shape:', df.shape)

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)

    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)

    df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

    X = df.drop('Churn', axis=1)
    y = df['Churn']

    X = pd.get_dummies(X, drop_first=True)
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    print(f'Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%')
    print(f'ROC-AUC:  {roc_auc_score(y_test, y_prob):.4f}')
    print(classification_report(y_test, y_pred))

    model_dir = os.path.join(os.path.dirname(__file__), '..', 'model')
    os.makedirs(model_dir, exist_ok=True)
    bundle = {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names
    }
    joblib.dump(bundle, os.path.join(model_dir, 'churn_model.pkl'))
    print('Model saved successfully to model/churn_model.pkl')

if __name__ == '__main__':
    train()
