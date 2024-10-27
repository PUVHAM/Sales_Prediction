# Sales_Prediction
This project focuses on predicting sales prices for advertising campaigns using Linear and Polynomial Regression models. The data used is based on various campaign attributes, including TV, Radio, Social Media spending, and Influencer types. The main goal is to forecast sales results based on these factors, allowing marketers to optimize their advertising budget.

<p align="center"> <img src="https://drive.google.com/uc?export=view&id=17FWo9xT2sHRMntRQpKVL9SBd3y75Z_96" width="590" height="355"/> </p>

## Table of Contents
- [Purpose](#purpose)
- [Project Structure](#project-structure)
- [Models Used](#models-used)
- [Usage](#usage)
  - [1. Setup_Conda_Environment](#1-setup-conda-environment)
  - [2. Run the Application](#2-run-the-application)
  - [3. Docker](#3-docker)
- [Dataset](#dataset)
- [Acknowledgments](#acknowledgments)

## Purpose
The purpose of this project is to predict sales based on advertisement expenditures across different channels and influencer categories. With this, inviduals/companies can make data-driven decisions to optimize their marketing budget allocation.

## Project Structure
The repository has the following structure:
- **frontend/data_analysis.py**: Handles data analysis and visualization for the frontend.
- **frontend/utils.py**: Contains utility functions for data processing in Streamlit.
- **backend/api/main.py**: Main FastAPI application, defines API endpoints.
- **src/config.py**: Configuration file for dataset paths and model settings.
- **src/load_dataset.py**: Loads and preprocesses the dataset.
- **src/train.py**: Contains training logic for Linear and Polynomial regression models.
- **src/models.py**: Defines custom model training functions.
- **src/schemas/**: Defines schemas for request and response formats in API.
- **docker/docker-compose.yml**: Defines services for running the app with Docker.
- **Dockerfile (frontend/ & backend/)**: Configuration file for building Docker images.
- **app.py**: Main Streamlit application file.
- **requirements.tx**t: List of project dependencies.

## Models Used
The project employs both Linear Regression and Polynomial Regression models for predicting sales based on the input attributes. The `train.py` script allows you to train these models and evaluate their performance.

## Usage

### 1. Setup Conda Environment

#### Prerequisites
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution)
  
1. **Clone the Repository:**
   
    ```bash
    git clone https://github.com/PUVHAM/Sales_Prediction.git
    cd Sales_Prediction
    ```

2. **Create and Activate Conda Environment:**

    ```bash
    conda create --name sales_prediction python=3.11
    conda activate sales_prediction
    ```

3. **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

### 2. Run the Application
#### Start FastAPI Backend
To run the backend on your local machine, use the following command:
```bash
uvicorn backend.api.main:app --reload
```
#### Start Streamlit Frontend
To run the frontend, which is built using Streamlit, execute the following:
```bash
streamlit run app.py
```
This will launch a web interface where you can:
- Train models using Linear or Polynomial Regression
- Upload data and predict sales based on input attributes
- View model performance and data visualizations

> **Note:** Ensure that the backend service is running at some point, as the frontend requires backend API services to function properly. Without the backend running, you may encounter connection errors when trying to use the application features.

### 3. Docker
#### Prerequisites
  - [Docker](https://www.docker.com/get-started): Make sure Docker is installed on your system.

To run the application using Docker Compose, you can use the following commands:
```bash
cd docker
docker-compose -p sales_prediction up --build -d
```
Once the containers are running, the application should be accessible at `http://localhost:8501` for the frontend and the backend API will be available at `http://localhost:8000`.

## Dataset
The dataset used in this project is `SalesPrediction.csv`, which contains information about advertising campaign expenditures and their corresponding sales figures. The file includes columns for TV, Radio, Social Media, Influencer type, and Sales outcome. To handle the categorical `Influencer` field, one-hot encoding is applied.

## Acknowledgments
- Dataset sourced from [here](https://drive.google.com/file/d/1A8kK0IEsT3w8htzU18ihFr5UV-euhquC/view).
- Special thanks to the creators of `FastAPI` and `streamlit` for the backend and frontend frameworks, and to the open-source community for valuable libraries like `scikit-learn`.

Feel free to reach out if you have any questions or issues!
