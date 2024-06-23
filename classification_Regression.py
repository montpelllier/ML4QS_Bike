import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from sklearn.model_selection import GridSearchCV

# Define paths and filenames
DATA_PATH = Path('./results/')
DATASET_FNAME = 'merged_dataset_result_feature.csv'
#RESULT_FNAME = 'merged_dataset_classification_result.csv'
# DATASET_FNAME = 'dataset4_result_feature.csv'
# RESULT_FNAME = 'classification_result4.csv'
#EXPORT_TREE_PATH = Path('./figures/classification/')

# Declare the number of features to select
N_FEATURE_SELECTION = 25

# Load the dataset
try:
    dataset = pd.read_csv(DATA_PATH / DATASET_FNAME, index_col=0)
except IOError as e:
    print('File not found, try to run previous crowdsignals scripts first!')
    raise e

dataset = dataset[1:]
# Separate features and target variables
X = dataset.drop(columns=['labelnormal', 'labelturnright', 'labelturnleft', 'labelbrake', 'labelstop', 'labelaccelerate'])
y = dataset[['labelnormal', 'labelturnright', 'labelturnleft', 'labelbrake', 'labelstop', 'labelaccelerate']].copy()
# Convert multi-label to single label
y['label'] = y.idxmax(axis=1)
label_encoder = LabelEncoder()
y['label'] = label_encoder.fit_transform(y['label'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y['label'], test_size=0.3, random_state=42)

# Handle class imbalance in the training set
train_data = X_train.copy()
train_data['label'] = y_train
train_data_balanced = pd.concat([resample(train_data[train_data['label'] == label],
                                          replace=True,  # Upsampling
                                          n_samples=train_data['label'].value_counts().max(),
                                          random_state=42) for label in train_data['label'].unique()])

X_train_balanced = train_data_balanced.drop(columns=['label'])
y_train_balanced = train_data_balanced['label']

# Define the Logistic Regression classifier
model = LogisticRegression(max_iter=1000)

# Perform Recursive Feature Elimination (RFE) on the balanced training set
rfe = RFE(model, n_features_to_select=N_FEATURE_SELECTION)
rfe.fit(X_train_balanced, y_train_balanced)

# Identify selected features
selected_features = X_train_balanced.columns[rfe.support_]
print("Selected Features:", selected_features)

# Transform training and testing sets to include only selected features
X_train_selected = X_train_balanced[selected_features]
X_test_selected = X_test[selected_features]

# Define parameter grid for Grid Search
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}

# Perform Grid Search for hyperparameter optimization
grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                           cv=3, n_jobs=-1, verbose=2, scoring='accuracy')

# Retrain the model with selected features and optimal parameters
grid_search.fit(X_train_selected, y_train_balanced)

# Output the best parameters found by Grid Search
print("Best Parameters:", grid_search.best_params_)

# Predict on the test set using the best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_selected)

# Convert predicted labels back to original format
y_pred_labels = label_encoder.inverse_transform(y_pred)
y_test_labels = label_encoder.inverse_transform(y_test)

# Evaluate model performance
accuracy = accuracy_score(y_test_labels, y_pred_labels)
print("Model Accuracy with Selected Features and Grid Search:", accuracy)
print("Classification Report:\n", classification_report(y_test_labels, y_pred_labels))
