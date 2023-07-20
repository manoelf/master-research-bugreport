from src.modules.util.util import Util as util
from src.modules.util.constant import Model, ModelName

class TainingModels:

    def __init__(self):
        pass


    def get_classification_artifacts(self, train, test):
        x_train, y_train = train.drop('label', axis=1), train['label']
        x_test, y_test = test.drop('label', axis=1), test['label']
        classes = train['label'].unique()
        
        return x_train, y_train, x_test, y_test, classes


    def get_model_metrics(self, model: Model, model_name: ModelName, train, test):
        x_train, y_train, x_test, y_test, classes = get_classification_artifacts(train, test)
        return util.get_metrics(model, model_name, x_train, y_train, x_test, y_test, classes)
