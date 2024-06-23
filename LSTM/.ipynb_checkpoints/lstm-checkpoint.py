{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "46aee271",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "\n",
    "from pathlib import Path\n",
    "\n",
    "from sklearn.model_selection import train_test_split, GridSearchCV\n",
    "from sklearn.metrics import accuracy_score, classification_report\n",
    "\n",
    "from imblearn.over_sampling import SMOTE\n",
    "from imblearn.under_sampling import RandomUnderSampler\n",
    "from imblearn.pipeline import make_pipeline\n",
    "\n",
    "from keras.wrappers.scikit_learn import KerasClassifier\n",
    "from keras.models import Sequential\n",
    "from keras.layers import LSTM, Dense, Dropout"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "93ab8686",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Define paths and filenames\n",
    "DATA_PATH = Path('../results/')\n",
    "DATASET_FNAME = 'dataset1_result_feature.csv'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "43111ee4",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load the dataset\n",
    "try:\n",
    "    dataset = pd.read_csv(DATA_PATH / DATASET_FNAME, index_col=0)\n",
    "except IOError as e:\n",
    "    print('File not found, try to run previous crowdsignals scripts first!')\n",
    "    raise e"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "8f3fe8d1",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Separate features and target variables\n",
    "X = dataset.drop(\n",
    "    columns=['labelnormal', 'labelturnright', 'labelturnleft', 'labelbrake', 'labelstop', 'labelaccelerate'])\n",
    "y = dataset[['labelnormal', 'labelturnright', 'labelturnleft', 'labelbrake', 'labelstop', 'labelaccelerate']].copy()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "10d0a249",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Convert multi-label to single label\n",
    "y_labels = np.argmax(y, axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "8ab00ac7",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Ensure the dataset is split in chronological order, assuming data is already sorted by time\n",
    "# Commented out original splitting by index to use train_test_split for randomized split\n",
    "# train_size = int(0.7 * len(dataset))  # Use 70% of the data for training\n",
    "# X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]\n",
    "# y_train, y_test = y['label'].iloc[:train_size], y['label'].iloc[train_size:]\n",
    "# print(len(X_train), len(X_test))\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y_labels, test_size=0.3, random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "9c34ac87",
   "metadata": {},
   "outputs": [],
   "source": [
    "pipeline = make_pipeline(\n",
    "    SMOTE(sampling_strategy='all', random_state=42, k_neighbors=5),\n",
    "    RandomUnderSampler(sampling_strategy='not minority', random_state=42)\n",
    ")\n",
    "\n",
    "# Use pipeline to resample\n",
    "X_train, y_train = pipeline.fit_resample(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "7f58aa52",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Convert DataFrames to numpy arrays\n",
    "X_train = np.array(X_train)\n",
    "X_test = np.array(X_test)\n",
    "\n",
    "# Reshape X_train and X_test for LSTM input\n",
    "X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])\n",
    "X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "892019c4",
   "metadata": {},
   "outputs": [],
   "source": [
    "def create_model(units, dropout_rate):\n",
    "    model = Sequential()\n",
    "    model.add(LSTM(units, input_shape=(X_train.shape[1], X_train.shape[2])))\n",
    "    model.add(Dropout(dropout_rate))\n",
    "    model.add(Dense(6, activation='softmax'))\n",
    "    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])\n",
    "    return model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "1a3d6062",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\zithe\\AppData\\Local\\Temp\\ipykernel_12564\\1900650505.py:2: DeprecationWarning: KerasClassifier is deprecated, use Sci-Keras (https://github.com/adriangb/scikeras) instead. See https://www.adriangb.com/scikeras/stable/migration.html for help migrating.\n",
      "  model = KerasClassifier(build_fn=create_model, epochs=10, batch_size=32)\n"
     ]
    }
   ],
   "source": [
    "# Create KerasClassifier for GridSearchCV\n",
    "model = KerasClassifier(build_fn=create_model, epochs=10, batch_size=32)\n",
    "# Define parameters grid\n",
    "param_grid = {\n",
    "    'units': [50, 100, 200],\n",
    "    'dropout_rate': [0.2, 0.3, 0.4]\n",
    "}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "fa7c1f3e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Fitting 3 folds for each of 9 candidates, totalling 27 fits\n",
      "Epoch 1/10\n",
      "165/165 [==============================] - 2s 3ms/step - loss: 1.1420 - accuracy: 0.6196\n",
      "Epoch 2/10\n",
      "165/165 [==============================] - 1s 3ms/step - loss: 0.4522 - accuracy: 0.8770\n",
      "Epoch 3/10\n",
      "165/165 [==============================] - 1s 3ms/step - loss: 0.2933 - accuracy: 0.9171\n",
      "Epoch 4/10\n",
      "165/165 [==============================] - 1s 3ms/step - loss: 0.2223 - accuracy: 0.9383\n",
      "Epoch 5/10\n",
      "165/165 [==============================] - 1s 3ms/step - loss: 0.1981 - accuracy: 0.9408\n",
      "Epoch 6/10\n",
      "165/165 [==============================] - 1s 3ms/step - loss: 0.1657 - accuracy: 0.9509\n",
      "Epoch 7/10\n",
      "165/165 [==============================] - 1s 3ms/step - loss: 0.1478 - accuracy: 0.9545\n",
      "Epoch 8/10\n",
      "165/165 [==============================] - 1s 3ms/step - loss: 0.1353 - accuracy: 0.9621\n",
      "Epoch 9/10\n",
      "165/165 [==============================] - 1s 3ms/step - loss: 0.1219 - accuracy: 0.9684\n",
      "Epoch 10/10\n",
      "165/165 [==============================] - 1s 4ms/step - loss: 0.1321 - accuracy: 0.9600\n",
      "Best: 0.125524 using {'dropout_rate': 0.2, 'units': 200}\n"
     ]
    }
   ],
   "source": [
    "# Grid search\n",
    "grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3, verbose=2)\n",
    "grid_result = grid.fit(X_train, y_train)\n",
    "# Print results\n",
    "print(\"Best: %f using %s\" % (grid_result.best_score_, grid_result.best_params_))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "12aed13b",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Get best model\n",
    "best_model = grid_result.best_estimator_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "2e0b4aaf",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "15/15 [==============================] - 0s 1ms/step\n"
     ]
    }
   ],
   "source": [
    "# Use model to predit\n",
    "y_pred = best_model.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "5cb9de5e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[1 0 0 0 0 0 0 0 0 0 0 0 4 0 0 0 0 0 0 0 0 0 0 0 0 3 0 5 0 0 0 0 3 0 0 0 0\n",
      " 0 0 0 0 0 1 0 0 0 0 0 0 4 0 0 0 0 4 3 0 5 0 0 0 0 0 2 2 0 0 0 2 5 2 0 0 0\n",
      " 0 2 0 0 5 0 0 0 0 3 0 0 0 0 0 0 0 0 0 0 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n",
      " 0 4 0 0 0 3 3 0 2 0 0 0 0 0 0 0 0 0 2 0 0 0 0 0 1 0 0 0 0 1 4 0 0 2 0 2 0\n",
      " 0 0 0 0 0 0 0 0 0 0 0 2 0 1 0 0 2 0 0 0 0 0 2 0 0 5 0 0 0 0 0 0 0 1 0 0 0\n",
      " 0 0 4 0 0 0 2 0 0 0 0 0 3 0 0 0 0 0 2 0 0 0 0 0 0 0 0 2 0 0 0 0 0 0 0 0 0\n",
      " 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0\n",
      " 0 1 4 0 0 0 0 2 0 0 0 5 0 1 0 5 0 0 0 0 0 0 2 0 0 0 0 0 2 0 0 0 2 0 0 0 0\n",
      " 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 0 0 0 0 0 0\n",
      " 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0\n",
      " 0 0 0 2 0 0 0 2 0 0 0 0 2 0 0 0 0 0 0 1 4 0 0 0 0 0 0 0 0 0 0 3 0 0 0 0 2\n",
      " 0 0 0 0 0 2 0 1 0 0 0 0 0 0 0 0 0 0 4 0 0 0 4 0 0 4 0 0 0 0 0 0 0 0 0 0 0\n",
      " 0 0 0 0 0 0 0 1 0 0 0 0 0 0]\n"
     ]
    }
   ],
   "source": [
    "print(y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "783a1f7f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model Accuracy with Selected Features and Grid Search: 0.9126637554585153\n",
      "Classification Report:\n",
      "               precision    recall  f1-score   support\n",
      "\n",
      "           0       0.98      0.92      0.95       389\n",
      "           1       0.83      0.77      0.80        13\n",
      "           2       0.74      0.97      0.84        29\n",
      "           3       0.40      0.75      0.52         8\n",
      "           4       0.79      0.92      0.85        12\n",
      "           5       0.38      0.71      0.50         7\n",
      "\n",
      "    accuracy                           0.91       458\n",
      "   macro avg       0.69      0.84      0.74       458\n",
      "weighted avg       0.93      0.91      0.92       458\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# Evaluate model performance\n",
    "accuracy = accuracy_score(y_test, y_pred)\n",
    "print(\"Model Accuracy with Selected Features and Grid Search:\", accuracy)\n",
    "print(\"Classification Report:\\n\", classification_report(y_test, y_pred))"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
