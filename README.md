# Diabetes Risk Predictor (Health-Risk-Predictor)

A simple end-to-end machine learning project that predicts diabetes risk for patients using the PIMA Indians Diabetes dataset, trained with RandomForest and deployed as a Streamlit app.

## 🚀 Project Overview

This repository contains everything needed to:
1. Preprocess the diabetes dataset
2. Train and evaluate a Random Forest classification model
3. Save and load the model and scaler
4. Run a friendly Streamlit UI for patient risk prediction

## 📁 Repository Structure

- `diabetes.csv` - Raw dataset
- `data_preprocessing.py` - Load dataset, split, standard scale, save arrays and scaler
- `data_model.py` - Train RandomForest, print metrics, save model pickle
- `app.py` - Streamlit app for interactive diabetes risk prediction
- `test-model.py` - Example to test predictions against saved test data
- `requirements.txt` - Python dependencies
- `README.md` - This file

## ✅ Features

- Data preprocessing with stratified train/test split
- Standard scaling pipeline saved as `scaler.pkl`
- RandomForest classifier with configurable hyperparameters
- Model metrics printed and confusion matrix saved
- Streamlit app with interactive patient input and probability/confidence outputs
- Prediction result display: high risk vs low risk

## 🛠️ Setup & Installation

1. Create and activate a Python environment (recommended):

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

2. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

3. Optional: If you hit binary compatibility issues (`numpy.dtype size changed`), reinstall numpy/pandas:

```bash
python -m pip install --upgrade --force-reinstall numpy pandas
```

## 🧠 How to Train the Model

Run preprocessing then model training:

```bash
python data_preprocessing.py
python data_model.py
```

This creates:
- `X_train.npy`, `X_test.npy`, `y_train.npy`, `y_test.npy`
- `scaler.pkl`
- `diabetes_model.pkl`
- `confusion_matrix.png`

## 🧪 How to Test the Model

Run the quick test script:

```bash
python test-model.py
```

> Note: `test-model.py` expects `X_test_raw.csv` and `y_test_raw.csv` (if exists). If these files are missing, use the prepared `X_test.npy/y_test.npy` from preprocessing and adapt the script.

## 🌐 Run the Streamlit App

Start the app:

```bash
streamlit run app.py
```

Open the local URL shown in terminal (`http://localhost:8501`). Input patient features and click `Analyze Risk`.

## 📌 Model Details

- Algorithm: RandomForestClassifier
- Number of features: 8
- Input features:
  - Pregnancies
  - Glucose
  - BloodPressure
  - SkinThickness
  - Insulin
  - BMI
  - DiabetesPedigreeFunction
  - Age

## 💡 Troubleshooting

- If you see import errors or binary incompatibility (e.g. expected dtype size changed), reinstall packages:

```bash
python -m pip uninstall -y numpy pandas
python -m pip install numpy pandas
```

- Ensure your active interpreter matches the environment where packages are installed.

Made with Python, scikit-learn, and Streamlit for intuitive health-risk prediction.
