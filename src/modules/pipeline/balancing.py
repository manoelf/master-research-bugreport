from imblearn.over_sampling import RandomOverSampler
import pandas as pd

class Balancing:

    def __init__(self):
        self.ros = RandomOverSampler()


    def oversample(self, data: pd.DataFrame, droped_feature='label', axis=1) -> pd.DataFrame:
        X = data.drop(droped_feature, axis=axis)
        x_ros, y_ros = self.ros.fit_resample(X, data[droped_feature])
        
        return pd.concat([x_ros, y_ros], axis=axis)
    