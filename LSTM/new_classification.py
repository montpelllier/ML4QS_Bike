import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.utils import resample
from keras.src.utils import to_categorical
from model import LSTMModel

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import make_pipeline
from collections import Counter

# Define paths and filenames
DATA_PATH = Path('../results/')
DATASET_FNAME = 'dataset1_result_feature.csv'

# Load the dataset
try:
    dataset = pd.read_csv(DATA_PATH / DATASET_FNAME, index_col=0)
except IOError as e:
    print('File not found, try to run previous crowdsignals scripts first!')
    raise e

# Separate features and target variables
X = dataset.drop(
    columns=['labelnormal', 'labelturnright', 'labelturnleft', 'labelbrake', 'labelstop', 'labelaccelerate'])
y = dataset[['labelnormal', 'labelturnright', 'labelturnleft', 'labelbrake', 'labelstop', 'labelaccelerate']].copy()

# Ensure the dataset is split in chronological order, assuming data is already sorted by time
# Commented out original splitting by index to use train_test_split for randomized split
# train_size = int(0.7 * len(dataset))  # Use 70% of the data for training
# X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
# y_train, y_test = y['label'].iloc[:train_size], y['label'].iloc[train_size:]
# print(len(X_train), len(X_test))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Use pipeline to resample
pipeline = make_pipeline(
    SMOTE(sampling_strategy='all', random_state=42, k_neighbors=5),  
    RandomUnderSampler(sampling_strategy='not minority', random_state=42)
)
X_train, y_train = pipeline.fit_resample(X_train, y_train.values)

# Convert DataFrames to numpy arrays
X_train = np.array(X_train)
X_test = np.array(X_test)

# Reshape X_train and X_test for LSTM input
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

# Define the model
model = LSTMModel(X_train, y_train)

# Train the model
model.train(epochs=10, batch_size=32)

# Use model to predit
y_pred = model.predict(X_test)

# Convert multi-label to single label
y_pred_labels = np.argmax(y_pred, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

# Evaluate model performance
accuracy = accuracy_score(y_test_labels, y_pred_labels)
print("Model Accuracy with Selected Features and Grid Search:", accuracy)
print("Classification Report:\n", classification_report(y_test_labels, y_pred_labels))