from keras.src.models import Sequential
from keras.src.layers.rnn.lstm import LSTM
from keras.src.layers.core.dense import Dense
from keras.src.layers.regularization.dropout import Dropout

class LSTMModel:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(LSTM(200, activation='relu', input_shape=(self.X_train.shape[1], self.X_train.shape[2])))
        model.add(Dropout(0.3))
        model.add(Dense(6, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def summary(self):
        self.model.summary()

    def train(self, epochs=10, batch_size=32):
        self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size)

    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def save(self, filename='lstm_model.h5'):
        self.model.save(filename)