import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import make_pipeline

from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

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

# Convert multi-label to single label
y_labels = np.argmax(y, axis=1)

# Ensure the dataset is split in chronological order, assuming data is already sorted by time
# Commented out original splitting by index to use train_test_split for randomized split
# train_size = int(0.7 * len(dataset))  # Use 70% of the data for training
# X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
# y_train, y_test = y['label'].iloc[:train_size], y['label'].iloc[train_size:]
# print(len(X_train), len(X_test))
X_train, X_test, y_train, y_test = train_test_split(X, y_labels, test_size=0.3, random_state=42)

pipeline = make_pipeline(
    SMOTE(sampling_strategy='all', random_state=42, k_neighbors=5),
    RandomUnderSampler(sampling_strategy='not minority', random_state=42)
)

# Use pipeline to resample
X_train, y_train = pipeline.fit_resample(X_train, y_train)

# Convert DataFrames to numpy arrays
X_train = np.array(X_train)
X_test = np.array(X_test)

# Reshape X_train and X_test for LSTM input
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

def create_model(units, dropout_rate):
    model = Sequential()
    model.add(LSTM(units, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(dropout_rate))
    model.add(Dense(6, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Create KerasClassifier for GridSearchCV
model = KerasClassifier(build_fn=create_model, epochs=10, batch_size=32)
# Define parameters grid
param_grid = {
    'units': [50, 100, 200],
    'dropout_rate': [0.2, 0.3, 0.4]
}

# Grid search
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3, verbose=2)
grid_result = grid.fit(X_train, y_train)
# Print results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# Get best model
best_model = grid_result.best_estimator_

# Use model to predit
y_pred = best_model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy with Selected Features and Grid Search:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))