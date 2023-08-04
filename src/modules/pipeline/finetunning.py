from src.modules.util.util import Util as util
from src.modules.util.constant import Model, ModelName
import pickle

class Finetunning:

    def __init__(self):
        pass


    def model_finetuning(self, model: Model, model_name: ModelName, x_train, y_train, x_test, y_test, classes, model_turned_metrics_path, model_pred_path, folds=10):
        model_tuned_metrics, model_pred = util.get_tuned_metrics(model, model_name, folds, x_train, y_train, x_test, y_test, classes)
        
        pickle.dump(model_tuned_metrics, open(model_turned_metrics_path, 'wb'))
        pickle.dump(model_pred, open(model_pred_path, 'wb'))

        return model_tuned_metrics, model_pred