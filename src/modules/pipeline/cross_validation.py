from numpy import mean, std
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold


class CrossValidation:

    def __init__(self):
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)


    def get_cross_validation_result(self, model, x_train, y_train):
        n_scores = cross_val_score(model, x_train, y_train, scoring='accuracy', cv=self.cv, n_jobs=-1, error_score='raise')
        return 'Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores))
