import os
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

class DatasetConfig:
    # Data directories
    DATASET_DIR = os.path.join(ROOT_DIR, 'data')
    RAW_DATA_DIR = os.path.join(DATASET_DIR, 'raw')
    PROCESSED_DATA_DIR = os.path.join(DATASET_DIR, 'processed')
    
    # Data files
    DATASET_PATH = os.path.join(RAW_DATA_DIR, 'SalesPrediction.csv')
    SCALER_PATH = os.path.join(PROCESSED_DATA_DIR, 'scaler.pkl')
    POLY_SCALER_PATH = os.path.join(PROCESSED_DATA_DIR, 'poly_scaler.pkl')
    RANDOM_SEED = 0
    TEST_SIZE = 0.33
    
class ModelConfig:
    MODEL_TYPES = {
        'Linear Regression': 'linear',
        'Polynomial Regression': 'poly',
        'Custom Polynomial Regression': 'custom_poly'
    }
    MODEL_DIR = os.path.join(ROOT_DIR, 'models')
    
    @classmethod
    def get_model_path(cls, model_type: str) -> str:
        return os.path.join(cls.MODEL_DIR, f"{model_type}_model.pkl")
    
class FeatureConfig:
    CATEGORICAL_FEATURES_OPTIONS = {
        'Influencer': ['Macro', 'Mega', 'Micro', 'Nano'],
    } 