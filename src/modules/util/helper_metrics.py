from sklearn.metrics import precision_score,  accuracy_score, recall_score, f1_score, roc_auc_score

class MetricsHelper:

    def __init__(self):
        pass
    

    def get_classification_artifacts(self, train, test, label='label'):
        x_train, y_train = train.drop(label, axis=1), train[label]
        x_test, y_test = test.drop(label, axis=1), test[label]
        classes = train[label].unique()
        
        return x_train, y_train, x_test, y_test, classes
    

    def compute_metrics(self, pred, y_test, average='weighted'):
        accuracy = accuracy_score(y_test, pred)
        precision = precision_score(y_test, pred, average=average)
        recall = recall_score(y_test, pred, average=average)
        f1 = f1_score(y_test, pred, average=average)
        auc = roc_auc_score(y_test, pred)

        return {"Metrics": ["Accuracy", "Precision", "Recall", "F1", "AUC"], "Scores": [accuracy, precision, recall, f1, auc]}
