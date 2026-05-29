import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Insurance Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =============================================================================
# LOAD MODEL + PREPROCESSING ARTIFACTS
# =============================================================================
@st.cache_resource
def load_artifacts():
    """Load all saved files once and cache them."""
    base_path = 'models'
    artifacts = {
        'model': joblib.load(os.path.join(base_path, 'final_model.pkl')),
        'scaler': joblib.load(os.path.join(base_path, 'scaler.pkl')),
        'feature_names': joblib.load(os.path.join(base_path, 'feature_names.pkl')),
        'scaling_cols': joblib.load(os.path.join(base_path, 'scaling_cols.pkl')),
        'encoding_maps': joblib.load(os.path.join(base_path, 'encoding_maps.pkl')),
    }
    return artifacts


# =============================================================================
# PREPROCESSING FUNCTION
# =============================================================================
def preprocess_input(input_dict, artifacts):
    """Convert user input into model-ready DataFrame."""
    df = pd.DataFrame([input_dict])
    
    # Convert dates
    for col in ['policy_start_date', 'policy_end_date', 'incident_date', 'claim_filed_date']:
        df[col] = pd.to_datetime(df[col])
    
    # --- Date features ---
    df['policy_age_days'] = (df['incident_date'] - df['policy_start_date']).dt.days
    df['claim_delay_days'] = (df['claim_filed_date'] - df['incident_date']).dt.days
    df['days_to_policy_expiry'] = (df['policy_end_date'] - df['incident_date']).dt.days
    df['incident_month'] = df['incident_date'].dt.month
    df['incident_weekday'] = df['incident_date'].dt.dayofweek
    df['is_weekend'] = df['incident_weekday'].isin([5, 6]).astype(int)
    df['incident_quarter'] = df['incident_date'].dt.quarter
    df['incident_year'] = df['incident_date'].dt.year
    df = df.drop(columns=['policy_start_date', 'policy_end_date', 
                          'incident_date', 'claim_filed_date'])
    
    # --- Ratio features ---
    df['claim_coverage_ratio'] = df['claim_amount'] / (df['coverage_amount'] + 1)
    df['claim_repair_ratio'] = df['claim_amount'] / (df['repair_estimate'] + 1)
    df['premium_coverage_ratio'] = df['premium_amount'] / (df['coverage_amount'] + 1)
    df['deductible_claim_ratio'] = df['deductible_amount'] / (df['claim_amount'] + 1)
    df['avg_previous_claim'] = df['total_claim_amount_previous'] / (df['previous_claims_count'] + 1)
    df['claim_vs_avg_ratio'] = df['claim_amount'] / (df['avg_previous_claim'] + 1)
    
    # --- Risk flags ---
    df['is_high_claim'] = (df['claim_amount'] > 100000).astype(int)
    df['claim_exceeds_coverage'] = (df['claim_amount'] > df['coverage_amount']).astype(int)
    df['no_police_report'] = (df['police_report_available'] == 'No').astype(int)
    df['no_witness'] = (df['witness_available'] == 'No').astype(int)
    df['late_filing'] = (df['claim_delay_days'] > 30).astype(int)
    df['odd_hour_incident'] = ((df['incident_hour'] >= 0) & (df['incident_hour'] <= 5)).astype(int)
    df['frequent_claimer'] = (df['claims_last_12_months'] > 2).astype(int)
    df['new_customer'] = (df['customer_tenure_years'] < 1).astype(int)
    df['recent_policy_change'] = (df['policy_changes_last_year'] > 0).astype(int)
    df['poor_documentation'] = (df['documentation_score'] < 5).astype(int)
    df['high_suspicion'] = (df['suspicious_activity_score'] > 7).astype(int)
    
    risk_flags = ['no_police_report', 'no_witness', 'late_filing', 'odd_hour_incident',
                  'frequent_claimer', 'new_customer', 'recent_policy_change',
                  'poor_documentation', 'high_suspicion', 'claim_exceeds_coverage']
    df['total_risk_flags'] = df[risk_flags].sum(axis=1)
    
    # --- Encoding ---
    enc = artifacts['encoding_maps']
    
    # Binary mapping
    for col in ['police_report_available', 'witness_available', 
                'third_party_involved', 'weekend_incident']:
        if col in df.columns:
            df[col] = df[col].map(enc['binary_map'])
    
    # Gender
    if 'gender' in df.columns:
        df['gender'] = df['gender'].map(enc['gender_map'])
    
    # Ordinal severity
    if 'incident_severity' in df.columns:
        df['incident_severity'] = df['incident_severity'].map(enc['severity_map'])
    
    # One-hot encoding for nominal columns
    nominal_cols = [c for c in enc['nominal_cols'] if c in df.columns]
    df = pd.get_dummies(df, columns=nominal_cols, drop_first=True)
    
    # Bool to int
    for col in df.select_dtypes(include='bool').columns:
        df[col] = df[col].astype(int)
    
    # Align columns with training features
    for col in artifacts['feature_names']:
        if col not in df.columns:
            df[col] = 0
    df = df[artifacts['feature_names']]
    
    # Apply scaler
    scaling_cols = [c for c in artifacts['scaling_cols'] if c in df.columns]
    df[scaling_cols] = artifacts['scaler'].transform(df[scaling_cols])
    
    return df


# =============================================================================
# RISK CATEGORY LOGIC
# =============================================================================
def get_risk_category(probability):
    if probability <= 0.25:
        return "🟢 Low Risk"
    elif probability <= 0.60:
        return "🟡 Medium Risk"
    else:
        return "🔴 High Risk"


# =============================================================================
# RECOMMENDATION ENGINE
# =============================================================================
def get_recommendations(probability, input_dict):
    recs = []
    
    # Risk-level base recommendation
    if probability <= 0.25:
        recs.append("✅ Process claim normally")
    elif probability <= 0.60:
        recs.append("⚠️ Verify documents before approval")
    else:
        recs.append("🚨 Send for manual fraud investigation")
    
    # Specific recommendations based on red flags
    claim_amt = input_dict['claim_amount']
    cov_amt = input_dict['coverage_amount']
    
    if claim_amt / (cov_amt + 1) > 0.8:
        recs.append("📋 High claim-to-coverage ratio - Manual claim review required")
    
    if input_dict['police_report_available'] == 'No':
        recs.append("📄 Missing police report - Request additional documents")
    
    if input_dict['claims_last_12_months'] > 2:
        recs.append("🔁 Multiple recent claims - Fraud investigation suggested")
    
    if input_dict['suspicious_activity_score'] > 7:
        recs.append("🚩 High suspicious activity score - Escalate to fraud team")
    
    if input_dict['documentation_score'] < 5:
        recs.append("📑 Poor documentation score - Ask for supporting documents")
    
    return recs


# =============================================================================
# KEY RISK FACTORS IDENTIFIER
# =============================================================================
def get_risk_factors(input_dict):
    factors = []
    
    ratio = input_dict['claim_amount'] / (input_dict['coverage_amount'] + 1)
    if ratio > 0.7:
        factors.append(f"High claim-to-coverage ratio: {ratio*100:.0f}%")
    
    if input_dict['police_report_available'] == 'No':
        factors.append("No police report filed")
    
    if input_dict['witness_available'] == 'No':
        factors.append("No witness available")
    
    if input_dict['claims_last_12_months'] > 2:
        factors.append(f"Frequent claimer: {input_dict['claims_last_12_months']} claims in 12 months")
    
    if input_dict['customer_tenure_years'] < 1:
        factors.append("New customer (less than 1 year tenure)")
    
    if input_dict['suspicious_activity_score'] > 7:
        factors.append(f"High suspicion score: {input_dict['suspicious_activity_score']}/10")
    
    if input_dict['documentation_score'] < 5:
        factors.append(f"Poor documentation: {input_dict['documentation_score']}/10")
    
    if 0 <= input_dict['incident_hour'] <= 5:
        factors.append(f"Odd hour incident ({input_dict['incident_hour']}:00)")
    
    delay = (pd.to_datetime(input_dict['claim_filed_date']) - 
             pd.to_datetime(input_dict['incident_date'])).days
    if delay > 30:
        factors.append(f"Late filing: {delay} days after incident")
    
    return factors if factors else ["No significant risk factors detected"]


# =============================================================================
# LOAD ARTIFACTS
# =============================================================================
try:
    artifacts = load_artifacts()
    model_loaded = True
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.info("Make sure you're running from project root and models/ folder exists")
    model_loaded = False


# =============================================================================
# HEADER
# =============================================================================
st.title("🛡️ Insurance Claim Fraud Detection")
st.markdown("**Enter claim details below to predict fraud probability**")


# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    **Model:** XGBoost Classifier  
    **Trained on:** 50,000+ claims  
    **Performance:** ROC-AUC 0.97+
    
    ---
    
    **Risk Categories:**
    - 🟢 Low: 0% - 25%
    - 🟡 Medium: 26% - 60%
    - 🔴 High: 61% - 100%
    
    ---
    
    **Instructions:**
    1. Fill all 4 tabs
    2. Click 'Analyze Claim'
    3. View fraud probability + risk + recommendations
    """)


# =============================================================================
# INPUT FORM
# =============================================================================
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "👤 Customer Profile",
    "📜 Policy Details",
    "🚗 Claim Details",
    "🚨 Incident & Risk"
])

# --- TAB 1: Customer ---
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        customer_age = st.number_input("Customer Age", 18, 100, 35)
        gender = st.selectbox("Gender", ['Male', 'Female'])
        occupation = st.selectbox("Occupation",
                                    ['Salaried', 'Self-Employed', 'Business', 'Retired', 'Other'])
    with col2:
        location_type = st.selectbox("Location Type", ['Urban', 'Rural', 'Semi-Urban'])
        customer_tenure_years = st.number_input("Customer Tenure (years)", 0.0, 30.0, 3.0, 0.5)

# --- TAB 2: Policy ---
with tab2:
    col1, col2 = st.columns(2)
    with col1:
        policy_type = st.selectbox("Policy Type", ['Auto', 'Health', 'Property'])
        policy_duration_years = st.number_input("Policy Duration (years)", 1, 30, 5)
        premium_amount = st.number_input("Premium Amount (₹)", 1000, 500000, 25000, 1000)
        coverage_amount = st.number_input("Coverage Amount (₹)", 10000, 10000000, 500000, 10000)
    with col2:
        deductible_amount = st.number_input("Deductible Amount (₹)", 0, 100000, 10000, 500)
        policy_changes_last_year = st.number_input("Policy Changes (last year)", 0, 10, 0)
        policy_start_date = st.date_input("Policy Start Date", pd.to_datetime('2022-01-01'))
        policy_end_date = st.date_input("Policy End Date", pd.to_datetime('2025-01-01'))

# --- TAB 3: Claim ---
with tab3:
    col1, col2 = st.columns(2)
    with col1:
        claim_amount = st.number_input("Claim Amount (₹)", 1000, 5000000, 50000, 1000)
        repair_estimate = st.number_input("Repair Estimate (₹)", 1000, 5000000, 45000, 1000)
        claim_type = st.selectbox("Claim Type",
                                    ['Accident', 'Medical', 'Theft', 'Fire', 'Natural Disaster'])
    with col2:
        incident_severity = st.selectbox("Incident Severity", ['Minor', 'Major', 'Total Loss'])
        incident_location = st.selectbox("Incident Location",
                                          ['Road', 'Hospital', 'Public Place', 'Home', 'Workplace'])
        incident_date = st.date_input("Incident Date", pd.to_datetime('2024-01-15'))
        claim_filed_date = st.date_input("Claim Filed Date", pd.to_datetime('2024-01-20'))

# --- TAB 4: Incident + Risk ---
with tab4:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Incident Details**")
        incident_hour = st.slider("Incident Hour (0-23)", 0, 23, 14)
        weekend_incident = st.selectbox("Weekend Incident?", ['Yes', 'No'])
        police_report_available = st.selectbox("Police Report Available?", ['Yes', 'No'])
        witness_available = st.selectbox("Witness Available?", ['Yes', 'No'])
        third_party_involved = st.selectbox("Third Party Involved?", ['Yes', 'No'])
    with col2:
        st.markdown("**Claim History & Risk Scores**")
        previous_claims_count = st.number_input("Previous Claims (lifetime)", 0, 20, 1)
        claims_last_12_months = st.number_input("Claims (last 12 months)", 0, 10, 0)
        total_claim_amount_previous = st.number_input("Total Previous Claim Amount (₹)",
                                                       0, 5000000, 30000, 1000)
        documentation_score = st.slider("Documentation Score (0-10)", 0, 10, 7)
        suspicious_activity_score = st.slider("Suspicious Activity Score (0-10)", 0, 10, 3)


# =============================================================================
# PREDICT BUTTON + OUTPUT
# =============================================================================
st.markdown("---")
predict_btn = st.button("🔮 Analyze Claim", type="primary", use_container_width=True)

if predict_btn and model_loaded:
    # Collect inputs
    input_dict = {
        'customer_age': customer_age, 'gender': gender, 'occupation': occupation,
        'location_type': location_type, 'customer_tenure_years': customer_tenure_years,
        'policy_type': policy_type, 'policy_duration_years': policy_duration_years,
        'premium_amount': premium_amount, 'coverage_amount': coverage_amount,
        'deductible_amount': deductible_amount,
        'policy_changes_last_year': policy_changes_last_year,
        'policy_start_date': policy_start_date, 'policy_end_date': policy_end_date,
        'claim_amount': claim_amount, 'repair_estimate': repair_estimate,
        'claim_type': claim_type, 'incident_severity': incident_severity,
        'incident_location': incident_location, 'incident_date': incident_date,
        'claim_filed_date': claim_filed_date, 'incident_hour': incident_hour,
        'weekend_incident': weekend_incident,
        'police_report_available': police_report_available,
        'witness_available': witness_available,
        'third_party_involved': third_party_involved,
        'previous_claims_count': previous_claims_count,
        'claims_last_12_months': claims_last_12_months,
        'total_claim_amount_previous': total_claim_amount_previous,
        'documentation_score': documentation_score,
        'suspicious_activity_score': suspicious_activity_score,
    }
    
    try:
        with st.spinner("Analyzing claim..."):
            # Preprocess and predict
            X = preprocess_input(input_dict, artifacts)
            probability = float(artifacts['model'].predict_proba(X)[0, 1])
            risk_category = get_risk_category(probability)
            recommendations = get_recommendations(probability, input_dict)
            risk_factors = get_risk_factors(input_dict)
        
        # ===== Display Results =====
        st.markdown("## 📊 Analysis Result")
        
        # Top metrics row
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Fraud Probability", f"{probability*100:.1f}%")
        with c2:
            st.metric("Risk Category", risk_category)
        with c3:
            decision = "❌ Suspect Fraud" if probability >= 0.5 else "✅ Likely Genuine"
            st.metric("Decision", decision)
        
        # Visual risk meter
        st.markdown("### 📈 Risk Meter")
        st.progress(probability)
        
        # Status alert (color-coded)
        if "Low" in risk_category:
            st.success(f"**Overall Assessment:** {risk_category}")
        elif "Medium" in risk_category:
            st.warning(f"**Overall Assessment:** {risk_category}")
        else:
            st.error(f"**Overall Assessment:** {risk_category}")
        
        # Two columns for factors and recommendations
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### 🚩 Key Risk Factors")
            for factor in risk_factors:
                st.markdown(f"- {factor}")
        with c2:
            st.markdown("### 💡 Suggested Recommendations")
            for rec in recommendations:
                st.markdown(f"- {rec}")
        
        # Expandable: full input details
        with st.expander("📋 View All Input Details"):
            st.json(input_dict)
    
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info("Check that all preprocessing artifacts are in models/ folder")


# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.caption("⚠️ This is an ML-based screening tool. Final decisions require human review.")

#venv\Scripts\activate
#streamlit run apps/streamlit_app.py