import pandas as pd
import altair as alt
import matplotlib as plt
import numpy as np
import seaborn as sns

from sklearn.metrics import precision_score, confusion_matrix, accuracy_score, recall_score, f1_score, plot_confusion_matrix
from sklearn import metrics
from sklearn.model_selection import GridSearchCV


class Util:

    def __init__(self):
        pass


    def get_data(self, path):
        return [pd.read_csv(path + "train.csv"), pd.read_csv(path + "test.csv")]


    def show_distribution_graph(self, df):
        df = df['label'].value_counts().to_frame()
        df = df.reset_index().rename(columns={"index": "Resolution", "label": "Total"})

        return alt.Chart(df).mark_bar().encode(
            x=alt.X('Resolution', sort='-y'),
            y='Total',
            color = alt.value("#ac97b4")
        )


    def model_confusion_matrix(self, model, x_test, y_test, classes):
        return plot_confusion_matrix(model, x_test, y_test, labels=classes, cmap=plt.cm.Blues, xticks_rotation = "vertical")


    def compute_metrics(self, pred, y_test):
        accuracy = accuracy_score(y_test, pred)
        precision = precision_score(y_test, pred, average='weighted')
        recall = recall_score(y_test, pred, average='weighted')
        f1 = f1_score(y_test, pred, average='weighted')

        return {"Metrics": ["Accuracy", "Precision", "Recall", "F1"], "Scores": [accuracy, precision, recall, f1]}


    def print_metrics(self, model_name, metrics):
        print(f"{model_name} Metrics:\n")
        for i in range(4):
            print(f"{metrics['Metrics'][i]} score is:\t{round(metrics['Scores'][i] * 100,2)}%")
        print("\n")


    def compute_metrics_per_class(self, pred, y_test):

        # Print the confusion matrix
        print(metrics.confusion_matrix(y_test, pred))

        # Print the precision and recall, among other metrics
        print(metrics.classification_report(y_test, pred, digits=2))


    def get_metrics(self, model, model_name, x_train, y_train, x_test, y_test, classes):
        model.fit(x_train, y_train)
        pred = model.predict(x_test)

        metrics = compute_metrics(pred,  y_test)
        print_metrics(model, model_name, metrics, x_test, y_test, classes)
        compute_metrics_per_class(pred, y_test)

        return metrics['Scores']


    def get_metric_data(self, models, models_names):
        # models: [NB_metrics,LG_metrics,DT_metrics,RF_metrics, GB_metrics]
        # TODO: fix this method
        metrics = np.array(models).flatten()
        metrics = list(map(lambda x: x*100, metrics))

        data = {"Metric":
                ['Accuracy', 'Precision', 'Recall', 'F1 Score'] * 5,
                "Metric Score": metrics,
                "Model":
                ['Naive Bayes'] * 4 +
                ['Logistic Regression'] * 4 +
                ['Decision Tree'] * 4 +
                ['Random Forest'] * 4 +
                ['Gradient Boosting'] * 4
            }

        return pd.DataFrame(data)


    def plot_metric_graph(self, data):
        g = sns.catplot(
            data=data,
            kind="bar", x="Metric", y="Metric Score", hue="Model",
            ci="sd", alpha=.6, height=6
        )
        g.set(ylim=(0, 100))
        g.despine(left=True)


    def get_params(self, model_name):
        if (model_name == "Logistic Regression"):
            return { 'C': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0] }
        elif (model_name == "Gradient Boosting"):
            return {
                'n_estimators': [50, 100],
                'max_depth': [3, 8],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 5],
                'max_features': [None, 5],
                'subsample': [0.5, 1]
                }
        else:
            return { 'max_depth': [None, 15, 35, 50], 'max_leaf_nodes': [None, 250, 500, 750, 1000, 5000]}


    def print_best_params(self, model_name, grid, folds):
        print(f'Hyperparams of {model_name}:\n')
        print(f'Got accuracy score of {grid.best_score_} in {folds}-fold')
        if (model_name == "Logistic Regression"):
            print(f'Best C: {grid.best_params_["C"]}')
        elif (model_name == "Gradient Boosting"):
            print(f'Best max depth: {grid.best_params_["max_depth"]}. Best number of estimators: {grid.best_params_["n_estimators"]}')
            print(f'Best min sample split: {grid.best_params_["min_samples_split"]}. Best min sample leaf: {grid.best_params_["min_samples_leaf"]}')
            print(f'Best max features: {grid.best_params_["max_features"]}. Best subsample: {grid.best_params_["subsample"]}')
        else:
            print(f'Best depth: {grid.best_params_["max_depth"]}. Best number of leafs: {grid.best_params_["max_leaf_nodes"]}')


    def get_tuned_metrics(self, model, model_name, folds, x_train, y_train, x_test, y_test, classes):
        params = self.get_params(model_name)

        grid = GridSearchCV(model, params, cv = folds)
        grid.fit(x_train, y_train)
        pred = grid.predict(x_test)

        self.print_best_params(model_name, grid, folds)
        metrics = self.compute_metrics(pred, y_test)
        self.print_metrics(grid, model_name, metrics, x_test, y_test, classes)
        return metrics['Scores'], pred
    
    
    def rename_column(self, data: pd.DataFrame, column_name: str, new_column_name: str):
        data[new_column_name] = data[column_name]
        data.drop(column_name, axis=1)
        