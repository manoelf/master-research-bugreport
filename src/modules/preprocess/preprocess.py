import pandas as pd
from tokenize import tokenize
from src.modules.util.constant import Features
from src.modules.util.util import Util


class Preprocess:

    def __init__(self):
        pass


    def fill_null_values(self, data: pd.DataFrame, feature: Features, inplace=True, axis=1):
        data[feature].fillna('', inplace=inplace)
        data[f'total_words_{feature}'] = data.apply(lambda row: len(list(tokenize(row[feature]))), axis=axis)
        
        return data
    

    def get_dummy_feature(self, data: pd.DataFrame):
        return pd.get_dummies(data)
    