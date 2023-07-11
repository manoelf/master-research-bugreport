import pandas as pd

class Data:

    def __init__(self):
        pass

    def fill_null_values(self, data: pd.DataFrame):

        data['description'].fillna('', inplace=True)
        data['total_words_desc'] = data.apply(lambda row: len(list(tokenize(row['description']))), axis=1)

        data['summary'].fillna('', inplace=True)
        data['total_words_summary'] = data.apply(lambda row: len(list(tokenize(row['summary']))), axis=1)

        return data


    def drop_feature(
            self,
            data: pd.DataFrame,
            features=["type", "flags", "assigned_to", "creator", "description", "summary", "id", "creation_time", "last_change_time", "resolution"]):
        
        data.drop(features, inplace=True, axis=1)
        # target_feature = data[["resolution"]]
        # data.drop("resolution", inplace=True, axis=1)

        return data
    

    def get_dummy_feature(self, data: pd.DataFrame):
        return pd.get_dummies(data)
    