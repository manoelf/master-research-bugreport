from src.modules.util.util import Util as util
from src.modules.util.constant import Model, ModelName

class TainingModels:

    def __init__(self):
        pass


    def train_model(self, model, x_train, y_train):
        model.fit(x_train, y_train)
        return model


    def predict_model(self, model, x_test):
        pred = model.predict(x_test)
        return pred


    def get_model_metrics(self, model: Model, model_name: ModelName, train, test):
        x_train, y_train, x_test, y_test, classes = util.get_classification_artifacts(train, test)
        return util.get_metrics(model, model_name, x_train, y_train, x_test, y_test, classes)
