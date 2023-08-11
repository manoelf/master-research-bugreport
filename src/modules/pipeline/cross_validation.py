from numpy import mean, std
from sklearn.model_selection import cross_val_score
from src.modules.util.constant import Model

class CrossValidation:

    def __init__(self):
        pass


    def get_cross_validation_result(self, model, x_train, y_train, scoring='accuracy', cv=Model.CV, n_jobs=-1, error_score='raise'):
        n_scores = cross_val_score(model, x_train, y_train, scoring=scoring, cv=cv, n_jobs=n_jobs, error_score=error_score)
        return '%s: %.3f (%.3f)' % (scoring, mean(n_scores), std(n_scores))

