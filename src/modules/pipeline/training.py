from src.modules.util.util import Util as util
from src.modules.util.constant import Model, ModelName

class TainingModels:

    def __init__(self):
        pass

    # def train_model(self, model, train_data, train_labels):
    #     model.fit(train_data, train_labels)
    #     return model

    def get_model_metrics(self, model: Model, model_name: ModelName, train, test):
        x_train, y_train, x_test, y_test, classes = get_classification_artifacts(train, test)
        return util.get_metrics(model, model_name, x_train, y_train, x_test, y_test, classes)
