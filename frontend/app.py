import streamlit as st
import pandas as pd
import requests
import os
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from fpdf import FPDF
import base64

# Streamlit config
st.set_page_config(page_title="Breast Cancer Prediction", layout="wide", page_icon="🎗️")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Dataset", "Visualizations", "Model Performance", "Prediction", "Batch Prediction", "Explainability", "About"])

st.sidebar.markdown("---")
st.sidebar.subheader("Model Info")
try:
    info_res = requests.get("http://localhost:8000/model-info")
    if info_res.status_code == 200:
        model_meta = info_res.json()
        st.sidebar.markdown("**Hybrid Architecture Active**")
        if "pre_diagnostic" in model_meta:
            st.sidebar.text(f"Pre-Diag Model: {model_meta['pre_diagnostic'].get('model_name')}")
        if "diagnostic_assessment" in model_meta:
            st.sidebar.text(f"Post-Diag Model: {model_meta['diagnostic_assessment'].get('model_name')}")
except Exception:
    st.sidebar.text("Model info unavailable")

# Utility to load image safely
def load_image(path):
    if os.path.exists(path):
        return Image.open(path)
    return None

def create_gauge_chart(probability):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Cancer Risk Probability (%)"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkred" if probability >= 0.5 else "darkgreen"},
            'steps': [
                {'range': [0, 50], 'color': "lightgreen"},
                {'range': [50, 75], 'color': "orange"},
                {'range': [75, 100], 'color': "red"}
            ]
        }
    ))
    return fig

def generate_pdf_report(patient_data, result):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=15, style='B')
    pdf.cell(200, 10, txt="Breast Cancer Risk Assessment Report", ln=1, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", size=12, style='B')
    pdf.cell(200, 10, txt="Patient Information:", ln=1)
    pdf.set_font("Arial", size=10)
    
    for key, value in patient_data.items():
        if value is not None:
            pdf.cell(200, 8, txt=f"{key}: {value}", ln=1)
        
    pdf.ln(10)
    pdf.set_font("Arial", size=12, style='B')
    pdf.cell(200, 10, txt="Prediction Result:", ln=1)
    pdf.set_font("Arial", size=10)
    
    pdf.cell(200, 8, txt=f"Engine Used: {result.get('model_type', 'N/A')}", ln=1)
    status = "High Risk (Malignant)" if result['prediction'] == 1 else "Low Risk (Benign)"
    pdf.cell(200, 8, txt=f"Risk Status: {status}", ln=1)
    pdf.cell(200, 8, txt=f"Probability: {result['probability']*100:.2f}%", ln=1)
    
    # Save to file
    pdf.output("report.pdf")
    return "report.pdf"

if page == "Home":
    st.title("🎗️ Breast Cancer Hybrid Prediction System")
    st.markdown("""
    Welcome to the Breast Cancer Prediction Dashboard!
    
    This application features a **Dual-Model Hybrid Architecture**:
    - **Pre-Test Risk Assessment**: Uses demographic and lifestyle factors to predict cancer risk *before* taking a mammogram or biopsy.
    - **Post-Test Diagnostic Classification**: Uses diagnostic features (Mammogram, Tumor Size, Lymph Nodes) to classify whether the tumor is benign or malignant.
    
    Use the navigation sidebar to explore the dataset or try the interactive prediction tool.
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
        st.subheader("Class Distribution (Pre-Diagnosis)")
        img_target = load_image('reports/figures/class_distribution_pie.png')
        if img_target:
            st.image(img_target, use_container_width=True)
            
    with col2:
        st.subheader("Dataset Balancings (SMOTENC)")
        img_sizes = load_image('reports/figures/dataset_sizes.png')
        if img_sizes:
            st.image(img_sizes, use_container_width=True)
            
    st.subheader("Feature Distributions by Target")
    img_box = load_image('reports/figures/boxplot_grid.png')
    if img_box:
        st.image(img_box, use_container_width=True)

elif page == "Model Performance":
    st.title("⚙️ Model Performance")
    st.markdown("Evaluation metrics and actual confusion matrices generated on the **held-out Test Set (15%)**.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Pre-Diagnostic Model (Risk)")
        img_cm_pre = load_image('reports/figures/pre_diagnostic_confusion_matrix.png')
        if img_cm_pre:
            st.image(img_cm_pre, use_container_width=True)
            
    with col2:
        st.subheader("Post-Diagnostic Model (Diagnostic)")
        img_cm_post = load_image('reports/figures/diagnostic_assessment_confusion_matrix.png')
        if img_cm_post:
            st.image(img_cm_post, use_container_width=True)

elif page == "Prediction":
    st.title("🔮 Hybrid Patient Prediction")
    st.markdown("Enter patient attributes to predict the risk of breast cancer. If diagnostic test results are provided, the system will automatically route to the highly accurate diagnostic classification model.")
    
    with st.form("prediction_form"):
        st.markdown("### Demographic & Lifestyle (Pre-Screening)")
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
            
            c1, c2 = st.columns(2)
            with c1:
                blood_pressure = st.number_input("Blood Pressure (Systolic)", min_value=80, max_value=200, value=120)
                cholesterol = st.number_input("Cholesterol", min_value=100, max_value=400, value=200)
            with c2:
                diabetes = st.selectbox("Diabetes", ["No", "Yes"])
                breastfeeding = st.selectbox("Breastfeeding History", ["No", "Yes", "Not Applicable"])

        with col3:
            exercise_days = st.number_input("Exercise Days/Week", min_value=0, max_value=7, value=3)
            annual_income = st.number_input("Annual Income (USD)", min_value=10000, max_value=500000, value=60000)
            
        st.markdown("---")
        st.markdown("### Clinical Diagnostics (Optional)")
        d1, d2, d3 = st.columns(3)
        with d1:
            mammogram = st.selectbox("Mammogram Result", ["Not Tested", "Normal", "Abnormal"])
        with d2:
            lymph_node = st.selectbox("Lymph Node Involvement", ["Not Tested", "No", "Yes"])
        with d3:
            tumor_size = st.number_input("Tumor Size (cm) (0 = Not Tested)", min_value=0.0, max_value=20.0, value=0.0)
            
        submit = st.form_submit_button("Predict Risk")

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
            "Blood_Pressure": blood_pressure,
            "Cholesterol": cholesterol,
            "Diabetes": diabetes,
            "Exercise_Days_Per_Week": exercise_days,
            "Breastfeeding_History": breastfeeding,
            "Annual_Income_USD": annual_income,
            "Mammogram_Result": mammogram if mammogram != "Not Tested" else None,
            "Lymph_Node_Involvement": lymph_node if lymph_node != "Not Tested" else None,
            "Tumor_Size_cm": tumor_size if tumor_size > 0.0 else None
        }
        
        try:
            response = requests.post("http://localhost:8000/predict", json=payload)
            if response.status_code == 200:
                result = response.json()
                
                # Display gauge chart and model used
                st.info(f"Engine Used: **{result.get('model_type', 'N/A')}**")
                st.plotly_chart(create_gauge_chart(result['probability']), use_container_width=True)
                
                if result['prediction'] == 1:
                    st.error(f"⚠️ High Risk Detected!")
                else:
                    st.success(f"✅ Low Risk.")
                
                if "shap_explanations" in result and result["shap_explanations"]:
                    st.subheader("Why this prediction?")
                    shap_vals = result["shap_explanations"]
                    top_features = dict(list(shap_vals.items())[:10])
                    
                    df_shap = pd.DataFrame(list(top_features.items()), columns=["Feature", "Impact"])
                    df_shap["Color"] = df_shap["Impact"].apply(lambda x: "red" if x > 0 else "blue")
                    
                    fig = px.bar(df_shap, x="Impact", y="Feature", orientation='h', color="Color", 
                                 color_discrete_map="identity", title="Top Contributing Features (SHAP)")
                    fig.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
                
                # Generate PDF report
                pdf_path = generate_pdf_report(payload, result)
                with open(pdf_path, "rb") as f:
                    pdf_bytes = f.read()
                st.download_button(label="📄 Download PDF Report",
                                   data=pdf_bytes,
                                   file_name="patient_report.pdf",
                                   mime="application/pdf")
            else:
                st.error(f"Error from API: {response.text}")
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to the backend API. Please make sure the FastAPI server is running.")

elif page == "Batch Prediction":
    st.title("📂 Batch CSV Prediction")
    st.markdown("Upload a CSV file containing patient attributes to get predictions. Diagnostic fields are optional.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df_batch = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(df_batch.head())
            
            if st.button("Run Batch Prediction"):
                # Replace NaNs with None for JSON payload
                df_batch = df_batch.replace({np.nan: None})
                patients_list = df_batch.to_dict('records')
                payload = {"patients": patients_list}
                
                response = requests.post("http://localhost:8000/predict_batch", json=payload)
                if response.status_code == 200:
                    results = response.json()['predictions']
                    
                    df_results = df_batch.copy()
                    df_results['Prediction'] = [res['prediction'] for res in results]
                    df_results['Cancer_Risk'] = [res['cancer_risk'] for res in results]
                    df_results['Probability'] = [res['probability'] for res in results]
                    df_results['Model_Type'] = [res.get('model_type', 'N/A') for res in results]
                    
                    st.success("Batch Prediction Complete!")
                    st.dataframe(df_results)
                    
                    csv = df_results.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name='batch_predictions.csv',
                        mime='text/csv',
                    )
                else:
                    st.error(f"Error from API: {response.text}")
        except Exception as e:
            st.error(f"Error processing file: {e}")

elif page == "Explainability":
    st.title("🔍 Explainable AI")
    st.markdown("Understand how the model makes decisions using SHAP (SHapley Additive exPlanations).")
    
    st.info("SHAP explanations are now generated dynamically for each patient prediction in the **Prediction** tab.")

elif page == "About":
    st.title("ℹ️ About the Project")
    st.markdown("""
    ### Hybrid Breast Cancer Prediction Clinical Decision Support System

    **Objective:** Build a highly accurate clinical decision-support system.

    #### Methodology
    1. **Dual-Model Architecture**: 
       - **Pre-Diagnostic Mode**: Drops post-facto clinical tests to predict pure risk.
       - **Post-Diagnostic Mode**: Incorporates Mammogram, Tumor Size, and Lymph Node status for highly accurate malignancy classification.
    2. **Data Pipeline**: 70/15/15 Data splits for Train/Validation/Test. SMOTENC is used strictly on the training set to resolve class imbalances safely.
    3. **Models**: XGBoost, Random Forest, and Logistic Regression are tuned via RandomizedSearchCV on a 3-fold cross-validation scheme.
    """)
    st.info("This system is designed for clinical decision support and should not replace professional medical advice.")
