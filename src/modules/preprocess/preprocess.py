import pandas as pd
from src.util.constant import Features


class Data:

    def __init__(self):
        pass


    def fill_null_values(self, data: pd.DataFrame, feature: Features):

        data[feature].fillna('', inplace=True)
        data['total_words_' + feature] = data.apply(lambda row: len(list(tokenize(row[feature]))), axis=1)
        
        # rename total_words_desc column in another function
        
        return data


    def drop_feature(
            self,
            data: pd.DataFrame,
            features=Features.features):
        
        data.drop(features, inplace=True, axis=1)

        return data
    

    def get_dummy_feature(self, data: pd.DataFrame):
        return pd.get_dummies(data)
    