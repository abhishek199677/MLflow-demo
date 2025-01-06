import os
import warnings
import sys
import logging
from urllib.parse import urlparse

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

import mlflow
import mlflow.sklearn
import dagshub

# Initialize DagsHub repository for tracking experiments
dagshub.init(repo_owner='puttapoguabhishek1007', repo_name='MLflow-demo', mlflow=True)




# Set up logging configuration
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def eval_metrics(actual, pred):
    """Evaluate metrics for model performance."""
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def main():
    # Suppress warnings and set random seed for reproducibility
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality dataset from the specified URL
    csv_url = "https://raw.githubusercontent.com/abhishek199677/MLflow-demo/refs/heads/main/WineQT.csv"
    
    try:
        data = pd.read_csv(csv_url, sep=",")
    except Exception as e:
        logger.exception("Unable to download training & test CSV. Check your internet connection. Error: %s", e)
        return

    print("Training data columns:", data.columns)

    # Split the data into training and test sets (75% train, 25% test)
    train, test = train_test_split(data)

    # Prepare features and target variable for training and testing
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    # Get hyperparameters from command line arguments or set defaults
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.2
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3

    with mlflow.start_run():
        # Train the ElasticNet model
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        # Make predictions on the test set
        predicted_qualities = lr.predict(test_x)

        # Evaluate model performance
        rmse, mae, r2 = eval_metrics(test_y, predicted_qualities)

        # Log metrics and parameters to MLflow
        print(f"ElasticNet model (alpha={alpha:.6f}, l1_ratio={l1_ratio:.6f}):")
        print(f"  RMSE: {rmse}")
        print(f"  MAE: {mae}")
        print(f"  R2: {r2}")

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        # Set remote server URI for DagsHub tracking
        remote_server_uri = "https://dagshub.com/puttapoguabhishek1007/MLflow-demo.mlflow"
        mlflow.set_tracking_uri(remote_server_uri)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Register the model if not using a local file store
        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(lr, "model", registered_model_name="ElasticnetWineModel")
        else:
            mlflow.sklearn.log_model(lr, "model")

if __name__ == "__main__":
    main()
