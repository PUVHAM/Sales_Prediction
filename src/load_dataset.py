import os
import joblib
import pandas as pd

from tqdm import tqdm
from src.config import DatasetConfig
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def download_dataset():
    import gdown
    file_id = '1A8kK0IEsT3w8htzU18ihFr5UV-euhquC'
    url = f'https://drive.google.com/uc?id={file_id}'
    
    os.makedirs(DatasetConfig.RAW_DATA_DIR, exist_ok=True)
    
    gdown.download(url,
                   output=DatasetConfig.DATASET_PATH,
                   quiet=True,
                   fuzzy=True)
    
def split_dataset(df):
    x = df[['TV', 'Radio', 'Social Media', 'Influencer_Macro', 'Influencer_Mega',
        'Influencer_Micro', 'Influencer_Nano']]
    y = df[['Sales']]
     
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                      test_size=DatasetConfig.TEST_SIZE,
                                                      random_state=DatasetConfig.RANDOM_SEED)
    
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    
    os.makedirs(DatasetConfig.PROCESSED_DATA_DIR, exist_ok=True)
    with tqdm(total=1, desc="Saving Scaler", bar_format="{l_bar}{bar} [ time left: {remaining} ]") as pbar:
        joblib.dump(scaler, DatasetConfig.SCALER_PATH)
        pbar.update(1)
    
    return x_train, y_train, x_test, y_test

def load_df(csv_path):
    if not os.path.exists(csv_path):
        try:
            download_dataset()
        except Exception as e:
            ERROR_MSG = 'Failed when attempting download the dataset. Please check the download process.'
            raise e(ERROR_MSG)
    df = pd.read_csv(csv_path)
    df = pd.get_dummies(df)
    df = df.fillna(df.mean())

    return df