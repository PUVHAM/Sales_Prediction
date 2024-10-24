import os
import streamlit as st
import requests
from src.config import ModelConfig, DatasetConfig
from frontend.utils import get_feature, preprocess_feature, configuration
from frontend.data_analysis import plot_figure

API_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
            
def create_directories():
    os.makedirs(DatasetConfig.RAW_DATA_DIR, exist_ok=True)
    os.makedirs(DatasetConfig.PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(ModelConfig.MODEL_DIR, exist_ok=True)

def predict_sales(input_data, model_option):
    """
    Function to predict sales.
    """
    if st.session_state.models_trained:
        try:
            selected_model_type = ModelConfig.MODEL_TYPES[model_option]
            prediction_request = {
                "model_type": selected_model_type,
                "input_data": preprocess_feature(input_data)
            }
            response = requests.post(f"{API_URL}/predict", json=prediction_request)
            
            if response.status_code == 200:
                prediction = response.json()["prediction"]
                st.success(f"Predicted Sales: **${prediction:,.2f}** 💵")
            else:
                st.error("Error making prediction. Please try again.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please train the models first using the 'Train Model' button in the sidebar.")

def display_model_performance(model_option):
    """
    Function to display model performance.
    """
    st.subheader("📊 Model Performance Metrics")

    if 'models_results' in st.session_state:
        selected_model_type = ModelConfig.MODEL_TYPES[model_option]

        if selected_model_type in st.session_state.models_results:
            model_metrics = st.session_state.models_results[selected_model_type]
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Training Accuracy", f"{model_metrics['train_accuracy']:.4f}")
            with col2:
                st.metric("Test Accuracy", f"{model_metrics['test_accuracy']:.4f}")
        else:
            st.warning("Model performance metrics not available yet. Please train the models.")
    else:
        st.warning("No models trained yet. Please train the models first.")

def main():
    st.set_page_config(
        page_title="Sales Prediction App",
        page_icon=":bar_chart:",
        layout="wide",
        menu_items={
            'Get Help': 'https://github.com/PUVHAM/sales_prediction',
            'Report a Bug': 'mailto:phamquangvu19082005@gmail.com',
            'About': "# Sales Prediction App\n"
                     "Predict sales using multiple machine learning models."
        }
    )
    st.title(':chart_with_upwards_trend: Sales Prediction App')
    
    st.markdown("""
    Welcome to the **Sales Prediction App**! This app allows you to predict sales using different machine learning models.
    You can explore different models and evaluate their performance.
    """)
    with st.sidebar:
        st.header("🛠️ Configuration")
        
        model_option = st.selectbox('Choose Model Type', ModelConfig.MODEL_TYPES.keys())

        train_models = st.button('Train Model', help="Train all models with current dataset")
        
        configuration(train_models, API_URL)
                    
    tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Model Performance", "📈 Data Analysis"])

    # Tab 1: Prediction
    with tab1:
        st.subheader("Enter Marketing Spend Information")

        input_data = get_feature()  

        if st.button('Predict 🎯'):
            predict_sales(input_data, model_option)
            

    # Tab 2: Model Performance
    with tab2:
        display_model_performance(model_option)
            
    # Tab 3: Data Analysis
    with tab3:
        plot_figure()

                        
if __name__ == "__main__":
    if 'models_trained' not in st.session_state:
        st.session_state.models_trained = False
        st.session_state.models_results = {}
    
    create_directories()
    main()