class DatasetConfig:
    DATASET_DIR = 'backend/data'
    SCALER_PATH = 'backend/data/scaler.pkl'
    POLY_SCALER_PATH = 'backend/data/poly_scaler.pkl'
    DATASET_PATH = 'backend/data/SalesPrediction.csv'
    RANDOM_SEED = 0
    TEST_SIZE = 0.33
    
class ModelConfig:
    MODEL_TYPES = {
        'Linear Regression': 'linear',
        'Polynomial Regression': 'poly',
        'Custom Polynomial Regression': 'custom_poly'
    }
    MODEL_DIR = 'backend/models'
    
class FeatureConfig:
    CATEGORICAL_FEATURES_OPTIONS = {
        'Influencer': ['Macro', 'Mega', 'Micro', 'Nano'],
    } 