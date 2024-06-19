import numpy as np
from sklearn.preprocessing import MinMaxScaler

class DataPreprocessor:
    def __init__(self):
        self.scaler = MinMaxScaler()

    def scale_data(self, data):
        return self.scaler.fit_transform(data)

    def inverse_transform_data(self, scaled_data, original_data):
        # This function assumes scaled_data is a 2D array with the same number of columns as original_data
        dummy = np.zeros((scaled_data.shape[0], original_data.shape[1]))
        dummy[:, 0] = scaled_data[:, 0]
        return self.scaler.inverse_transform(dummy)[:, 0]

    def create_time_steps(self, data, time_step=5):
        X, y = [], []
        for i in range(len(data) - time_step):
            X.append(data[i:(i + time_step), :])
            y.append(data[i + time_step, 0])
        return np.array(X), np.array(y)
    
    def create_dataset(self, dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - time_step - 1):
            a = dataset[i:(i + time_step), :]
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return np.array(dataX), np.array(dataY)