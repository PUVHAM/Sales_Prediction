import os
import joblib
import streamlit as st
from tqdm import tqdm
from fastapi import HTTPException
from src.models import create_polynomial_features
from src.config import DatasetConfig, ModelConfig
from src.load_dataset import load_df, split_dataset
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

class SalesModel:    
    def __init__(self, model_type='linear', degree=2):
        self.degree = degree
        self.model_type = model_type
        self.model = LinearRegression()
        self.poly_features = {
            'linear': None,
            'poly': PolynomialFeatures(degree=self.degree, interaction_only=False),
            'custom_poly': None
        }
        
        self.poly = 'custom_poly' if model_type == 'custom_poly' else self.poly_features.get(self.model_type)

        self.input_shape = None

    def train(self, x_train, y_train):
        self.input_shape = x_train.shape[1]
        
        if self.poly == 'custom_poly':
            x_train = create_polynomial_features(x_train, degree=self.degree)
        elif self.poly:
            x_train = self.poly.fit_transform(x_train)
            joblib.dump(self.poly, DatasetConfig.POLY_SCALER_PATH)
            
        print('Start training...')
        self.model.fit(x_train, y_train)
        print('Training completed!')
        
        self.transformed_shape = x_train.shape[1]
        
    def save(self, path):
        """Save the model and its metadata"""
        model_data = {
            'model': self.model,
            'input_shape': self.input_shape,
            'transformed_shape': getattr(self, 'transformed_shape', None),
            'model_type': self.model_type,
            'degree': self.degree
        }
        joblib.dump(model_data, path)
        
    @classmethod
    def load(cls, path):
        """Load the model and its metadata"""
        model_data = joblib.load(path)
        instance = cls(model_type=model_data['model_type'], degree=model_data['degree'])
        instance.model = model_data['model']
        instance.input_shape = model_data['input_shape']
        instance.transformed_shape = model_data['transformed_shape']
        return instance
        
    def evaluate(self, x_train, y_train, x_test, y_test):
        if self.poly == 'custom_poly':
            x_train = create_polynomial_features(x_train, degree=self.degree)
            x_test = create_polynomial_features(x_test, degree=self.degree)
        elif self.poly:
            x_train = self.poly.transform(x_train)
            x_test = self.poly.transform(x_test)
            
        y_pred_test = self.model.predict(x_test)
        y_pred_train = self.model.predict(x_train)
        
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        
        return r2_train, r2_test
    
    def predict(self, input_data):
        if input_data.shape[1] != self.input_shape:
            raise ValueError(f"Input data has {input_data.shape[1]} features, but model was trained with {self.input_shape} features")
            
        if self.poly == 'custom_poly':
            input_data = create_polynomial_features(input_data, degree=self.degree)
        elif self.poly:
            self.poly = joblib.load(DatasetConfig.POLY_SCALER_PATH)
            input_data = self.poly.transform(input_data)

        prediction = self.model.predict(input_data)
        return prediction

@st.cache_data(max_entries=1000, ttl=3600)
def train_and_cache_models(model_type):
    df = load_df(DatasetConfig.DATASET_PATH)
    x_train, y_train, x_test, y_test = split_dataset(df)

    model = SalesModel(model_type)
    model.train(x_train, y_train)
    train_accuracy, test_accuracy = model.evaluate(x_train, y_train, x_test, y_test)
   
    # Save the model
    model.save(ModelConfig.get_model_path(model_type))
    
    results = {
        'model': model.model.__class__.__name__,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy
    }
    return model, results

@st.cache_data(max_entries=1000)
def train_models_parallel(model_types):
    models = {}
    results = {}

    with ThreadPoolExecutor() as executor:
        future_to_model = {
            executor.submit(train_and_cache_models, model_type): model_type for model_type in model_types
        }

        for future in tqdm(as_completed(future_to_model), total=len(model_types), desc="Training models"):
            model_type = future_to_model[future]
            try:
                model, result = future.result()
                models[model_type] = model
                results[model_type] = result
            except Exception as e:
                print(f"{model_type} generated an error: {e}")
                raise HTTPException(status_code=500, detail="Error in training models.")

    return models, results