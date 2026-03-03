# Configuration file for the demand forecasting project

# General settings
RANDOM_SEED = 42

# Dataset paths
DATASET_1_PATH = "demand-forecasting-kernels-only (1)/train.csv"
DATASET_2_PATH = "hierarchical_sales_data.csv"
DATASET_3_PATH = "rossmann-store-sales (2)/train ross.csv"
STORE_INFO_PATH = "rossmann-store-sales (2)/store ross.csv"

# Model settings
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# Output settings
MODEL_SAVE_PATH = "models/"
RESULTS_SAVE_PATH = "results/"