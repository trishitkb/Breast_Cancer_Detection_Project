# Technology Selection and Strategies

This document highlights the reasoning behind the technologies chosen for the Breast Cancer Hybrid Prediction System (BC-HPS), outlining the alternatives considered and why certain practices were avoided.

## 1. Application Framework Stack

### Why FastAPI?
- **Speed & Async Support**: FastAPI is built on Starlette and Pydantic, offering Node.js-level performance. This is crucial for serving ML models where inference latency is a bottleneck.
- **Data Validation**: Instead of manually parsing JSON dictionaries, FastAPI relies on Pydantic schemas (e.g., `PatientData` and `BatchPatientData`). This enforces strict data types (e.g., Age must be an int between 18 and 100), meaning the ML model never receives bad data.
- **Alternatives Avoided**: Flask and Django. Flask lacks built-in async support and automatic OpenAPI documentation. Django is too heavy and monolithic for a simple inference microservice.

### Why Streamlit?
- **Data-Centric UI**: Streamlit provides rapid UI development natively in Python, making it perfect for data science applications.
- **Widget States**: Streamlit handles forms, sliders, and session states without requiring JavaScript, Redux, or heavy frontend state management.
- **Alternatives Avoided**: React, Vue, or Angular. While these offer ultimate customizability, they require a separate JavaScript codebase and significantly increase the development time for what is essentially a data dashboard and form intake application.

## 2. Machine Learning Algorithm Strategy

### Why Random Forest Classifier?
- **Non-linear Capabilities**: Medical data often contains non-linear interactions between variables (e.g., the interplay between age, hormone therapy, and tumor size). Random Forests capture these intrinsically.
- **Robustness to Outliers**: Unlike logistic regression, Random Forests are mostly immune to outliers in the dataset.
- **Feature Importance**: Random Forests inherently provide Gini importance metrics, which were critical during the initial exploratory data analysis to prune useless features.
- **Alternatives Avoided**: Deep Neural Networks (DNNs). DNNs are overkill for tabular data of 10,000 rows. They require significantly more hyperparameter tuning, are prone to overfitting on small tabular sets, and act as complete "black boxes," which is unacceptable in clinical tools.

## 3. Explanability and Transparency

### Why SHAP (SHapley Additive exPlanations)?
- **Local Interpretability**: We needed a way to explain a *single* patient's prediction, not just the model's global behavior. SHAP calculates the exact marginal contribution of a specific patient's feature (e.g., "This patient's BMI increased risk by 12%").
- **Clinical Trust**: Medical professionals require actionable insights. By using SHAP, we transform a binary output ("High Risk") into a transparent reasoning chain.
- **Alternatives Avoided**: LIME (Local Interpretable Model-agnostic Explanations). While LIME is faster, it relies on local surrogate models that can be unstable. SHAP provides mathematically consistent allocations of feature importance.

## 4. Architectural Patterns Avoided

### Avoided: The Monolithic Notebook Script
Many ML projects remain as Jupyter notebooks or a single Python file that runs the model and prints the output.
- **Why Avoided**: Notebooks are entirely unscalable and cannot be used by end-users (clinicians). We decoupled the project into a strict Client-Server structure to simulate a production-grade enterprise application.

### Avoided: Automated Blind Retraining
Initially, there was a script (`retrain.py`) intended to automatically retrain the model and overwrite the `.pkl` files.
- **Why Avoided**: In a medical context, models cannot be blindly retrained and deployed without rigorous manual clinical validation. The automated script was removed to prevent accidental overwrites of the carefully tuned dual-pipeline.
