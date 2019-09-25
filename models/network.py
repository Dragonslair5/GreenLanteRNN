from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

class Rnn:

    def __init__(self, length_of_sequence=100, in_out_neurons=100, n_hidden=10, batch_size=32, epoch=100, model_load=False):

        if model_load == True:
            return
        
        self.length_of_sequence = length_of_sequence
        self.in_out_neurons = in_out_neurons
        self.n_hidden = n_hidden

        model = Sequential()
        model.add(LSTM(n_hidden, batch_input_shape=(None, length_of_sequence, 1), return_sequences=False))
        model.add(Dense(in_out_neurons))
        model.add(Activation("linear"))
        optimizer = Adam(lr=0.001)
        model.compile(loss="mean_squared_error", optimizer=optimizer)

        self.model = model
        self.batch_size = batch_size
        self.epoch = epoch

    def train(self, train_data, valid_data):
        early_stopping = EarlyStopping(monitor='val_loss', mode='auto', patience=20)
        self.model.fit(train_data, valid_data,
            batch_size=self.batch_size,
            epochs=self.epoch,
            validation_split=0.1,
            callbacks=[early_stopping]
        )

    def predict(self, data):
        return self.model.predict(data)

    def save(self, file_name, include_optimizer=False):
        self.model.save(file_name, include_optimizer=include_optimizer)

    def load(self, file_name, compile=False):
        self.model = load_model(file_name, compile=compile)