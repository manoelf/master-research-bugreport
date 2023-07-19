from imblearn.over_sampling import RandomOverSampler
import pandas as pd

class Balancing:

    def __init__(self):
        self.ros = RandomOverSampler()

    
    def oversample(self, data: pd.DataFrame):
        X = data.drop('label', axis=1)
        X_ros, y_ros = self.ros.fit_resample(X, data['label'])

        return data