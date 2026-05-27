import os
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

os.makedirs('models', exist_ok=True)

df = pd.read_csv('data/customer_churn_business_dataset.csv')

# Normalize payment method to match UI options
df['payment_method'] = df['payment_method'].replace({'Card': 'Credit Card'})

# Feature engineering (mirrors preprocess_input in app.py)
df['revenue_per_month'] = df['total_revenue'] / df['tenure_months']
df['is_inactive'] = (df['last_login_days_ago'] > 14).astype(int)
df['has_complaint'] = (df['complaint_type'] != 'No Complaint').astype(int)
df['support_intensity'] = df['support_tickets'] / (df['tenure_months'] + 1)
df['low_engagement'] = (df['email_open_rate'] < 0.3).astype(int)
df['is_high_value'] = (df['total_revenue'] > 1500).astype(int)

# Label encode binary columns
label_encoders = {}
for col in ['gender', 'discount_applied', 'price_increase_last_3m']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# One-hot encode multi-class columns
multiclass_cols = ['country', 'city', 'customer_segment', 'signup_channel',
                   'contract_type', 'payment_method', 'complaint_type', 'survey_response']
df = pd.get_dummies(df, columns=multiclass_cols)

# Drop non-feature columns
df = df.drop(columns=['customer_id'])

target = 'churn'
X = df.drop(columns=[target])
y = df[target]

feature_columns = list(X.columns)

# Scale numerical columns
scaled_columns = ['age', 'tenure_months', 'monthly_logins', 'weekly_active_days',
                  'avg_session_time', 'features_used', 'usage_growth_rate',
                  'last_login_days_ago', 'monthly_fee', 'total_revenue',
                  'payment_failures', 'support_tickets', 'avg_resolution_time',
                  'csat_score', 'escalations', 'email_open_rate',
                  'marketing_click_rate', 'nps_score', 'referral_count',
                  'revenue_per_month', 'support_intensity']

# Keep only scaled_columns that exist in X
scaled_columns = [c for c in scaled_columns if c in X.columns]

scaler = StandardScaler()
X[scaled_columns] = scaler.fit_transform(X[scaled_columns])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                      eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)

score = model.score(X_test, y_test)
print(f"Test accuracy: {score:.4f}")

with open('models/xgboost_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('models/label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)
with open('models/feature_columns.pkl', 'wb') as f:
    pickle.dump(feature_columns, f)
with open('models/scaled_columns.pkl', 'wb') as f:
    pickle.dump(scaled_columns, f)

print("All artifacts saved to models/")

#launch command: venv\Scripts\streamlit.exe run app/app.py