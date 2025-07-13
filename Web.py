import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import xgboost as xgb
import matplotlib.pyplot as plt

# 1. Load model
model_xgb = joblib.load('xgb.pkl')

# 2. Configure SHAP explainer
feature_label = ['IC_DeepFeature1147', 'Zeff_DeepFeature1540', 'VMI_exponential_glrlm_RunVariance', 
                 'Zeff_wavelet-LHL_glszm_LargeAreaHighGrayLevelEmphasis', 'IC_gradient_firstorder_Kurtosis', 
                 'VMI_exponential_firstorder_Kurtosis', 'IC_DeepFeature288', 'Zeff_DeepFeature309', 
                 'VMI_DeepFeature1645', 'IC_DeepFeature599', 'Zeff_DeepFeature928', 'VMI_DeepFeature1109', 
                 'PEI_DeepFeature1021', 'VMI_DeepFeature1817', 'VMI_logarithm_glszm_SmallAreaLowGrayLevelEmphasis', 
                 'IC_DeepFeature1877', 'PEI_DeepFeature1218', 'Zeff_wavelet-HHL_gldm_DependenceVariance', 
                 'IC_original_glszm_GrayLevelNonUniformity', 'Zeff_wavelet-HLL_glszm_SizeZoneNonUniformityNormalized', 
                 'IC_wavelet-LLH_ngtdm_Coarseness', 'Zeff_DeepFeature763', 'IC_DeepFeature1130', 'IC_DeepFeature1791']

# 3. Streamlit input
st.title('Web Predictor for Post-CAS NIILs')
st.sidebar.header('Input Features')

# Input feature form
inputs = {}
for feature in feature_label:
    inputs[feature] = st.sidebar.number_input(feature, min_value=-10.0, max_value=10.0, value=0.0)

# Convert input values into a Pandas DataFrame
input_df = pd.DataFrame([inputs])

# 4. Prediction button
if st.sidebar.button('Predict'):
    try:
        # Ensure correct input data
        input_data = xgb.DMatrix(input_df)  # Pass DataFrame format data directly without .values
        prediction = model_xgb.predict(input_data)[0]  # Make prediction

        # Display prediction result
        st.subheader('Predicted Possibility of Post-CAS NIILs')
        st.write(f'Predicted Value: {prediction}')

        # Compute SHAP values
        explainer = shap.TreeExplainer(model_xgb)
        shap_values = explainer.shap_values(input_df)

        # 5. Display SHAP force plot
        st.subheader('SHAP Force Plot')
        shap.initjs()
        shap.force_plot(explainer.expected_value, shap_values[0], input_df.iloc[0, :], feature_names=feature_label, matplotlib=True, contribution_threshold=0.1)
        plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)

        st.image("shap_force_plot.png")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")