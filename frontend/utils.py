import requests
import streamlit as st
from src.config import FeatureConfig

def get_feature_input(feature_name, feature_type):
    label = f"{feature_name}"    
    if feature_type == 'Float':
        return st.number_input(label, value=0.0)
    
    elif feature_type == 'Categorical':
        options = FeatureConfig.CATEGORICAL_FEATURES_OPTIONS.get(feature_name)
        if options:
            return st.selectbox(label, options)
        else:
            return st.text_input(label)
        
def get_feature():
    # Define feature information
    feature_info = [
        ('TV Spend', 'Float'),
        ('Radio Spend', 'Float'),
        ('Social Media Spend', 'Float'),
        ('Influencer', 'Categorical')
    ]

    input_data = {}
    col1, col2 = st.columns(2)
    
    for i, (feature_name, feature_type) in enumerate(feature_info):
        with col1 if i % 2 == 0 else col2:
            input_data[feature_name] = get_feature_input(feature_name, feature_type)
            
    return input_data

def preprocess_feature(input_data):
    influencer_category = input_data.pop("Influencer", None)  
    if influencer_category:
        for category in FeatureConfig.CATEGORICAL_FEATURES_OPTIONS["Influencer"]:
            input_data[f"Influencer_{category}"] = 1 if influencer_category == category else 0
            
    return input_data

def configuration(train_models, api_url):    
    if train_models:
        with st.spinner('Training model...'):  
            response = requests.post(f"{api_url}/train_model/")
            if response.status_code == 200:
                st.session_state.models_trained = True
                st.session_state.models_results = response.json()["result"]
                st.success('✅ Training completed successfully!')
            else:
                st.error("❌ Error in training models.")