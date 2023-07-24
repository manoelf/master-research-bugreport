import pandas as pd
from src.modules.util.constant import Model

class Balancing:

    def __init__(self):
        pass


    def oversample(self, data: pd.DataFrame, droped_feature='label', axis=1) -> pd.DataFrame:
        X = data.drop(droped_feature, axis=axis)
        x_ros, y_ros = Model.ROS.fit_resample(X, data[droped_feature])
        
        return pd.concat([x_ros, y_ros], axis=axis)
    