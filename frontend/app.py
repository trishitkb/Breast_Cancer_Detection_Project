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
        st.sidebar.text(f"Model: {model_meta.get('model_name', 'N/A')}")
        st.sidebar.text(f"Version: {model_meta.get('dataset_version', 'v1.0')}")
        st.sidebar.text(f"Balancing: {model_meta.get('dataset_balancing', 'N/A')}")
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
        pdf.cell(200, 8, txt=f"{key}: {value}", ln=1)
        
    pdf.ln(10)
    pdf.set_font("Arial", size=12, style='B')
    pdf.cell(200, 10, txt="Prediction Result:", ln=1)
    pdf.set_font("Arial", size=10)
    
    status = "High Risk (Malignant)" if result['prediction'] == 1 else "Low Risk (Benign)"
    pdf.cell(200, 8, txt=f"Risk Status: {status}", ln=1)
    pdf.cell(200, 8, txt=f"Probability: {result['probability']*100:.2f}%", ln=1)
    
    # Save to file
    pdf.output("report.pdf")
    return "report.pdf"

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
            
    st.subheader("Feature Distributions (Violin Plots)")
    img_violin = load_image('reports/figures/violin_plots.png')
    if img_violin:
        st.image(img_violin, use_container_width=True)
    else:
        st.info("Image not available. Run the EDA script.")

elif page == "Model Performance":
    st.title("⚙️ Model Performance")
    st.markdown("Evaluation metrics and comparison across different models and dataset balancings.")
    
    st.subheader("ROC Curve Comparison")
    img_roc = load_image('reports/figures/roc_curve_comparison.png')
    if img_roc:
        st.image(img_roc, use_container_width=True)
        
    st.subheader("Precision-Recall Curve Comparison")
    img_pr = load_image('reports/figures/pr_curve_comparison.png')
    if img_pr:
        st.image(img_pr, use_container_width=True)
        
    st.subheader("Confusion Matrices (Best Models by Dataset)")
    img_cm = load_image('reports/figures/confusion_matrices.png')
    if img_cm:
        st.image(img_cm, use_container_width=True)

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
                
                # Display gauge chart
                st.plotly_chart(create_gauge_chart(result['probability']), use_container_width=True)
                
                if result['prediction'] == 1:
                    st.error(f"⚠️ High Risk Detected!")
                else:
                    st.success(f"✅ Low Risk.")
                
                if "shap_explanations" in result and result["shap_explanations"]:
                    st.subheader("Why this prediction?")
                    st.markdown("These are the most important features contributing to this patient's prediction. "
                                "Red bars push the risk higher, blue bars push the risk lower.")
                    
                    shap_vals = result["shap_explanations"]
                    # Take top 10 features
                    top_features = dict(list(shap_vals.items())[:10])
                    
                    df_shap = pd.DataFrame(list(top_features.items()), columns=["Feature", "Impact"])
                    df_shap["Color"] = df_shap["Impact"].apply(lambda x: "red" if x > 0 else "blue")
                    
                    fig = px.bar(df_shap, x="Impact", y="Feature", orientation='h', color="Color", 
                                 color_discrete_map="identity", title="Top Contributing Features")
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
    st.markdown("Upload a CSV file containing patient attributes to get predictions for multiple patients at once.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df_batch = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(df_batch.head())
            
            if st.button("Run Batch Prediction"):
                # Convert DataFrame to list of dicts
                patients_list = df_batch.to_dict('records')
                payload = {"patients": patients_list}
                
                response = requests.post("http://localhost:8000/predict_batch", json=payload)
                if response.status_code == 200:
                    results = response.json()['predictions']
                    
                    # Combine original data with predictions
                    df_results = df_batch.copy()
                    df_results['Prediction'] = [res['prediction'] for res in results]
                    df_results['Cancer_Risk'] = [res['cancer_risk'] for res in results]
                    df_results['Probability'] = [res['probability'] for res in results]
                    
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

elif page == "About":
    st.title("ℹ️ About the Project")
    st.markdown("""
    ### Breast Cancer Prediction Clinical Decision Support System

    **Objective:** Build a clinical decision-support system for breast cancer prediction with a strict focus on minimizing false negatives (missed malignant tumors).

    #### Methodology
    1. **Data Pipeline**: Python, Pandas, Scikit-learn for handling missing values and preprocessing. Imbalanced classes (80:20 Benign:Malignant) were managed using SMOTE (Synthetic Minority Over-sampling Technique).
    2. **Model Training**: A robust ML pipeline evaluating Logistic Regression, Random Forest, and Gradient Boosting over 5-Fold Cross Validation.
    3. **Threshold Optimization**: The prediction probability threshold was optimized based on the F2-score to heavily penalize False Negatives, rather than relying on default 0.5 accuracy metrics.
    4. **Explainability**: SHAP (SHapley Additive exPlanations) is deeply integrated to explain the "Why" behind individual and global predictions, ensuring clinical transparency.
    5. **Architecture**: 
       - **Backend**: FastAPI
       - **Frontend**: Streamlit
    """)
    st.info("This system is designed for clinical decision support and should not replace professional medical advice.")

