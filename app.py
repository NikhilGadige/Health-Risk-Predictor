import streamlit as st
import pickle
import pandas as pd

@st.cache_resource
def load_files():
    with open('diabetes_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

st.markdown(""" <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    </style>""", unsafe_allow_html=True)

st.markdown('<p class="main-header">Diabetes Risk Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Health Assessment Tool</p>', unsafe_allow_html=True)

try:
    model, scaler = load_files()
except:
    st.error("Model files not found! Run data_preprocessing.py and data_model.py first.")
    st.stop()

tab1, tab2 = st.tabs(["Patient Assessment", "About the Model"])

with tab1:
    st.markdown("### Enter Patient Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("Personal Info")
        age = st.number_input("Age", min_value=21, max_value=81, value=33, step=1)
        pregnancies = st.selectbox("Number of Pregnancies", options=list(range(0, 18)), index=3)
        
    with col2:
        st.markdown("Vital Signs")
        glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, max_value=200, value=120, step=1)
        bp = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=122, value=70, step=1)
        
    with col3:
        st.markdown("Body Measurements")
        bmi = st.number_input("BMI (Body Mass Index)", min_value=0.0, max_value=67.0, value=32.0, step=0.1, format="%.1f")
        skin = st.slider("Skin Thickness (mm)", 0, 99, 20)
    
    col4, col5 = st.columns(2)
    
    with col4:
        insulin = st.slider("Insulin Level (μU/mL)", 0, 846, 79, help="Measure of insulin in blood")
        
    with col5:
        dpf = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5, 0.01,help="Genetic predisposition score")
    
    st.markdown("<br>", unsafe_allow_html=True)
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        predict_btn = st.button("Analyze Risk", type="primary", use_container_width=True)
    
    if predict_btn:
        columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness','Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        
        input_df = pd.DataFrame([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]], columns=columns)
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]

        st.markdown("---")
        st.markdown("### Assessment Results")

        result_col1, result_col2, result_col3 = st.columns(3)
        
        with result_col1:
            if prediction == 1:
                st.error("High Risk")
                st.markdown("##### **Status:** Diabetes Likely")
            else:
                st.success("Low Risk")
                st.markdown("##### **Status:** No Diabetes")
        
        with result_col2:
            st.metric("Diabetes Probability", f"{probability[1]*100:.1f}%")
        
        with result_col3:
            confidence = max(probability) * 100
            if confidence > 80:
                confidence_label = "Very High"
            elif confidence > 70:
                confidence_label = "High"
            elif confidence > 60:
                confidence_label = "Moderate"
            else:
                confidence_label = "Low"
            st.metric("Model Confidence", f"{confidence:.1f}% ({confidence_label})")

        st.markdown("### Probability Breakdown")
        
        prob_col1, prob_col2 = st.columns(2)
        with prob_col1:
            st.markdown(f"**No Diabetes:** {probability[0]*100:.1f}%")
            st.progress(probability[0])
        with prob_col2:
            st.markdown(f"**Diabetes:** {probability[1]*100:.1f}%")
            st.progress(probability[1])

        with st.expander("View Detailed Input Summary"):
            st.dataframe(input_df, use_container_width=True, hide_index=True)
        
with tab2:
    st.markdown("### About This Application")
    
    info_col1, info_col2 = st.columns(2)
    
    with info_col1:
        st.markdown("""
        **Model Information**
        - **Algorithm:** Random Forest Classifier
        - **Accuracy:** ~75.97%
        - **Dataset:** PIMA Indians Diabetes Database
        - **Training Samples:** 614 patients
        - **Testing Samples:** 154 patients
        """)
    
    with info_col2:
        st.markdown("""
        **Input Features**
        - Number of Pregnancies
        - Glucose Level (mg/dL)
        - Blood Pressure (mm Hg)
        - Skin Thickness (mm)
        - Insulin Level (μU/mL)
        - BMI (Body Mass Index)
        - Diabetes Pedigree Function
        - Age (years)
        """)
    
    st.markdown("---")
    

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2382/2382533.png", width=100)
    st.markdown("### Quick Stats")
    st.metric("Model Accuracy", "75.97%")
    st.metric("Dataset Size", "768 patients")
    st.metric("Features Used", "8")
    
    st.markdown("---")
    st.markdown("### How to Use")
    st.markdown("""
    1. Enter patient information
    2. Click 'Analyze Risk'
    3. View prediction results
    4. Check detailed metrics
    """)
    
