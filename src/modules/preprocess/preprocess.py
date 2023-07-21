import pandas as pd
from tokenize import tokenize
from src.util.constant import Features


class Preprocess:

    def __init__(self):
        pass


    def fill_null_values(self, data: pd.DataFrame, feature: Features, inplace=True, axis=1):
        if feature == Features.description:
            data = Features.rename_column(data, 'total_words_desc', 'total_words_description')
            
        data[feature].fillna('', inplace=inplace)
        data['total_words_' + feature] = data.apply(lambda row: len(list(tokenize(row[feature]))), axis=axis)
        
        return data
    

    def get_dummy_feature(self, data: pd.DataFrame):
        return pd.get_dummies(data)
    