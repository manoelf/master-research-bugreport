from sklearn.metrics import precision_score,  accuracy_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize

class MetricsHelper:

    def __init__(self):
        pass
    

    def get_classification_artifacts(self, train, test, label='label'):
        x_train, y_train = train.drop(label, axis=1), train[label]
        x_test, y_test = test.drop(label, axis=1), test[label]
        classes = train[label].unique()
        
        return x_train, y_train, x_test, y_test, classes
    

    def compute_metrics(self, pred, y_test, predict_prob, average='weighted'):
        
        accuracy = accuracy_score(y_test, pred)
        precision = precision_score(y_test, pred, average=average, zero_division=0)
        recall = recall_score(y_test, pred, average=average)
        f1 = f1_score(y_test, pred, average=average)
        
        classes = list(y_test)
        y_true = label_binarize(y_test, classes=classes)
        
        y_scores = predict_prob.reshape(-1, 1)
        y_true = y_true.reshape(-1, len(classes))
        
        auc = roc_auc_score(y_true, y_scores, average=None)

        return {"Metrics": ["Accuracy", "Precision", "Recall", "F1", "AUC"], "Scores": [accuracy, precision, recall, f1, auc[0]]}

    