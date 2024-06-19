import tensorflow as tf

class ModelTrainer:
    def build_model(self, input_shape):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.LSTM(50, return_sequences=True, input_shape=input_shape))
        model.add(tf.keras.layers.LSTM(50, return_sequences=False))
        model.add(tf.keras.layers.Dense(25))
        model.add(tf.keras.layers.Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def train_model(self, model, X, y):
        model.fit(X, y, batch_size=1, epochs=1)
        model.save('stock_predictor.h5')
