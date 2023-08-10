import pandas as pd
import altair as alt
import numpy as np
import seaborn as sns
import time
import csv

from collections import defaultdict
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from src.modules.util.constant import ModelName as mn, DataShowMetrics as ds, Features
from src.modules.pipeline.training import TainingModels as tm
from src.modules.util.helper_metrics import MetricsHelper as mh


class Util:

    def __init__(self):

        default_params = { 
                'max_depth': [None, 15, 35, 50], 
                'max_leaf_nodes': [None, 250, 500, 750, 1000, 5000]
            }
        
        self.model_params = defaultdict(lambda: default_params)
        self.model_params[mn.LG] = {'C': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]}
        self.model_params[mn.GB] = {
                'n_estimators': [50, 100],
                'max_depth': [3, 8],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 5],
                'max_features': [None, 5],
                'subsample': [0.5, 1]
            }
        

    def get_data(self, path):
        return [pd.read_csv(path + "train.csv"), pd.read_csv(path + "test.csv")]


    def get_params(self, model_name):
        return self.model_params[model_name]
    

    def rename_column(self, data: pd.DataFrame, column_name: str, new_column_name: str):
        data[new_column_name] = data[column_name]
        data.drop(column_name, axis=1)
        
        return data


    def get_metrics(self, model, model_name, x_train, y_train, x_test, y_test):
        initial_time = time.time()
        
        model = tm().train_model(model, x_train, y_train)
        pred = tm().predict_model(model, x_test)
        pred = pd.DataFrame(pred).iloc[:-6]

        metrics = mh().compute_metrics(pred, y_test)
        
        final_time = time.time() - initial_time

        ShowMetrics().show_metrics(model, model_name, metrics, x_test, y_test)
        ShowMetrics().plot_model_confusion_matrix(y_test, pred)
        print(f'time: {final_time}')
        
        return metrics['Scores'], final_time


    def get_models_metric_data(self, models):
        # models: [NB_metrics, LG_metrics, DT_metrics, RF_metrics, GB_metrics]
        metrics = np.array(models).flatten()
        metrics = list(map(lambda x: x*100, metrics))

        data = {"Metric": ['Accuracy', 'Precision', 'Recall', 'F1 Score'] * 5,
                "Metric Score": metrics,
                "Model":
                [mn.NB] * 4 +
                [mn.LG] * 4 +
                [mn.DT] * 4 +
                [mn.RF] * 4 +
                [mn.GB] * 4
            }

        return pd.DataFrame(data)
    

    def get_tuned_metrics(self, model, model_name, folds, x_train, y_train, x_test, y_test, classes):
        params = self.get_params(model_name)

        grid = GridSearchCV(model, params, cv=folds)
        grid = tm.train_model(x_train, y_train)
        pred = tm.predict_model(x_test)

        ShowMetrics.show_best_params(model_name, grid, folds)
        metrics = mh.compute_metrics(pred, y_test)
        ShowMetrics.show_metrics(grid, model_name, metrics, x_test, y_test, classes)

        return metrics['Scores'], pred
    

    def remove_features(self, data, features=Features.train_test_features, inplace=False, axis=1):
        data = data.drop(features, inplace=inplace, axis=axis)
        return data
    
    
    def save_result(self, metrics, time, model_name, file_name='result.csv'):
        accuracy, precision, recall, f1, auc = metrics[0], metrics[1], metrics[2], metrics[3], metrics[4] 
        model_data = [accuracy, precision, recall, f1, auc, time]
        path = 'src/data/' + model_name + '/' + file_name

        with open(path, mode='a', newline='') as csv_file:
            escritor_csv = csv.writer(csv_file, delimiter=';')
            escritor_csv.writerow(model_data)



class ShowMetrics:

    def __init__(self):
        pass


    def show_distribution_graph(self, df, columns=ds.columns, x_label='Resolution', y_label='Total', color_value="#ac97b4"):
        df = df['label'].value_counts().to_frame()
        df = df.reset_index().rename(columns=columns)

        return alt.Chart(df).mark_bar().encode(x=alt.X(x_label, sort='-y'), y=y_label, color = alt.value(color_value))


    def plot_model_confusion_matrix(self, y_test, pred, classes):
        cm = confusion_matrix(y_test, pred)
        cm_array_df = pd.DataFrame(
            cm,
            index=classes,
            columns=classes
        )
        sns.heatmap(cm_array_df, annot=True, cmap=sns.cm.rocket_r, fmt="d")


    def show_best_params(self, model_name, grid, folds):
        print(f'Hyperparams of {model_name}: \nGot accuracy score of {grid.best_score_} in {folds}-fold')
        if (model_name == mn.LG): print(f'Best C: {grid.best_params_["C"]}')
        elif (model_name == mn.GB):
            print(f'Best max depth: {grid.best_params_["max_depth"]}. Best number of estimators: {grid.best_params_["n_estimators"]}')
            print(f'Best min sample split: {grid.best_params_["min_samples_split"]}. Best min sample leaf: {grid.best_params_["min_samples_leaf"]}')
            print(f'Best max features: {grid.best_params_["max_features"]}. Best subsample: {grid.best_params_["subsample"]}')
        else:
            print(f'Best depth: {grid.best_params_["max_depth"]}. Best number of leafs: {grid.best_params_["max_leaf_nodes"]}')


    def show_metrics(self, model_name, metrics):
        print(f"{model_name} Metrics:\n")
        for i in range(4):
            print(f"{metrics['Metrics'][i]} score is:\t{round(metrics['Scores'][i] * 100,2)}%")
        print("\n")


    def show_metrics_per_class(self, pred, y_test, classes, digits=2):
        # return the confusion matrix and the precision and recall, among other metrics
        return confusion_matrix(y_test, pred, classes), classification_report(y_test, pred, digits=digits)


    def plot_metric_graph(self, data, kind='bar', x="Metric", y="Metric Score", hue="Model", ci="sd", alpha=.6, height=6, x_ylim=0, y_ylim=100, left=True):
        g = sns.catplot(data=data,kind=kind, x=x, y=y, hue=hue, ci=ci, alpha=alpha, height=height)
        g.set(ylim=(x_ylim, y_ylim))
        g.despine(left=left)
