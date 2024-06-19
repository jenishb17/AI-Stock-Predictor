import tensorflow as tf

class ModelUpdater:
    def load_model(self):
        return tf.keras.models.load_model('stock_predictor.h5')

    def update_model(self, X, y):
        model = self.load_model()
        model.fit(X, y, batch_size=1, epochs=1)
        model.save('stock_predictor.h5')

    def predict(self, model, X):
        return model.predict(X)
