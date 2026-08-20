import streamlit as st
import pandas as pd
import requests
import os
from PIL import Image

# Streamlit config
st.set_page_config(page_title="Breast Cancer Prediction", layout="wide", page_icon="🎗️")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Dataset", "Visualizations", "Model Performance", "Prediction", "Explainability"])

# Utility to load image safely
def load_image(path):
    if os.path.exists(path):
        return Image.open(path)
    return None

if page == "Home":
    st.title("🎗️ Breast Cancer Prediction Project")
    st.markdown("""
    Welcome to the Breast Cancer Prediction Dashboard!
    
    This application is an end-to-end Machine Learning pipeline that predicts whether a breast tumor is malignant or benign based on patient attributes.
    
    ### Objectives:
    - Maximize recall for malignant cases.
    - Provide data-driven diagnosis support.
    - Explain predictions to build clinical trust.
    
    Use the navigation sidebar to explore the dataset, view EDA visualizations, see the model performance, or try out the interactive prediction tool.
    """)

elif page == "Dataset":
    st.title("📊 Dataset Overview")
    st.markdown("This section provides a preview and statistics of the raw dataset used to train the models.")
    
    try:
        df = pd.read_csv('data/raw/breast_cancer_prediction.csv')
        st.write(f"**Dataset Shape:** {df.shape[0]} rows, {df.shape[1]} columns")
        
        st.subheader("Data Preview")
        st.dataframe(df.head(100))
        
        st.subheader("Statistical Summary")
        st.write(df.describe())
    except FileNotFoundError:
        st.error("Dataset not found. Please run the data generation scripts first.")

elif page == "Visualizations":
    st.title("📈 Exploratory Data Analysis")
    st.markdown("Insights and visualizations derived from the dataset.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Target Distribution")
        img_target = load_image('reports/figures/target_distribution.png')
        if img_target:
            st.image(img_target, use_container_width=True)
        else:
            st.info("Image not available. Run the EDA script.")
            
    with col2:
        st.subheader("Correlation Heatmap")
        img_heatmap = load_image('reports/figures/correlation_heatmap.png')
        if img_heatmap:
            st.image(img_heatmap, use_container_width=True)
        else:
            st.info("Image not available. Run the EDA script.")

elif page == "Model Performance":
    st.title("⚙️ Model Performance")
    st.markdown("Evaluation metrics for the trained models.")
    
    # Normally we might load this from a saved JSON, but for now we'll just write placeholders or 
    # expect the user to run the scripts and see the console output.
    st.info("The Best Model was Random Forest. Run the notebook/scripts to see full comparison.")
    st.markdown("""
    - **High Recall**: Optimized to minimize false negatives.
    - **SMOTE**: Handled class imbalance effectively.
    """)

elif page == "Prediction":
    st.title("🔮 Patient Prediction")
    st.markdown("Enter patient attributes to predict the risk of breast cancer.")
    
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=45)
            gender = st.selectbox("Gender", ["Female", "Male"])
            bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
            family_history = st.selectbox("Family History", ["No", "Yes"])
            smoking = st.selectbox("Smoking", ["No", "Yes"])
            alcohol = st.selectbox("Alcohol Consumption", ["No", "Yes"])
            phys_activity = st.selectbox("Physical Activity", ["Low", "Moderate", "High"])
            
        with col2:
            hormone_therapy = st.selectbox("Hormone Therapy", ["No", "Yes"])
            menopause_status = st.selectbox("Menopause Status", ["Pre", "Post", "Not Applicable"])
            genetic_mutation = st.selectbox("Genetic Mutation", ["Negative", "Positive"])
            tumor_size = st.number_input("Tumor Size (cm)", min_value=0.0, max_value=10.0, value=2.5)
            lymph_node = st.selectbox("Lymph Node Involvement", ["No", "Yes"])
            mammogram = st.selectbox("Mammogram Result", ["Normal", "Suspicious"])

        with col3:
            bp = st.number_input("Blood Pressure", min_value=80, max_value=200, value=120)
            chol = st.number_input("Cholesterol", min_value=100, max_value=400, value=200)
            diabetes = st.selectbox("Diabetes", ["No", "Yes"])
            exercise = st.number_input("Exercise Days/Week", min_value=0, max_value=7, value=3)
            breastfeeding = st.selectbox("Breastfeeding History", ["Yes", "No", "Not Applicable"])
            income = st.number_input("Annual Income USD", min_value=0, max_value=500000, value=60000)
            
        submit = st.form_submit_button("Predict")

    if submit:
        payload = {
            "Age": age,
            "Gender": gender,
            "BMI": bmi,
            "Family_History": family_history,
            "Smoking": smoking,
            "Alcohol_Consumption": alcohol,
            "Physical_Activity": phys_activity,
            "Hormone_Therapy": hormone_therapy,
            "Menopause_Status": menopause_status,
            "Genetic_Mutation": genetic_mutation,
            "Tumor_Size_cm": tumor_size,
            "Lymph_Node_Involvement": lymph_node,
            "Mammogram_Result": mammogram,
            "Blood_Pressure": bp,
            "Cholesterol": chol,
            "Diabetes": diabetes,
            "Exercise_Days_Per_Week": exercise,
            "Breastfeeding_History": breastfeeding,
            "Annual_Income_USD": income
        }
        
        try:
            response = requests.post("http://localhost:8000/predict", json=payload)
            if response.status_code == 200:
                result = response.json()
                if result['prediction'] == 1:
                    st.error(f"⚠️ High Risk Detected! (Probability: {result['probability']:.2f})")
                else:
                    st.success(f"✅ Low Risk. (Probability: {1 - result['probability']:.2f})")
            else:
                st.error(f"Error from API: {response.text}")
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to the backend API. Please make sure the FastAPI server is running.")

elif page == "Explainability":
    st.title("🔍 Explainable AI")
    st.markdown("Understand how the model makes decisions using SHAP (SHapley Additive exPlanations).")
    
    st.subheader("Feature Importance")
    img_fi = load_image('reports/figures/feature_importance.png')
    if img_fi:
        st.image(img_fi, use_container_width=True)
    else:
        st.info("Feature importance plot not available.")
        
    st.subheader("SHAP Summary Plot")
    st.markdown("This plot shows the most important features and how they push the prediction higher or lower.")
    img_shap = load_image('reports/figures/shap_summary.png')
    if img_shap:
        st.image(img_shap, use_container_width=True)
    else:
        st.info("SHAP plot not available.")

