import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    roc_curve,
    auc,
    roc_auc_score,
    precision_recall_curve,
    make_scorer,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    precision_recall_curve,
)

from sklearn.ensemble import RandomForestClassifier
import shap

shap.initjs()

import warnings

warnings.filterwarnings("ignore")

class interpret_model:
    """
    interpret_model is a class to explain the execution and result in RandomForestClassifier.
    Can be use to plot the ROC Curve, confusion_matrix, Shap summary_plot.

    We include a summary function for you to get the similar result as Shikumika customer profiling function.
    However, we add a couple function for you to dive deep on different part of model explainability.

    Parameters
    ----------
    X : Features for prediction
    y: Prediction labels
    params: list of parameters for model
    test_size: Between 0 to 1. The proportion of test size of the data you provided
    model_tuning: Boolean. If you want to tune your model parameters
    scorer: make_scorer for model tuning
    threshold: Between 0 to 1. The threshold to decide either positive or negative.
    Examples
    --------
    >>> from data_marketing.cprofiling import model
    >>> model = model.interpret_model(X,y)
    """

    def __init__(
        self,
        data,
        model,

    ):

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(data)

        self.shap_values = shap_values

    def new_summary_plot(self, columns_to_show=None, num_samples=5000):

        """
        The interpret_model our team use to put in our notebook.

        The function give you a thought on how the model preform and the model explaination of how features effect predictions.
        This is neccesary to run before other functions.

        Parameters
        ----------
        columns_to_show : The column you want to include in Shap summary_plot
        num_samples : The number of data you would like to use to caluculate shap value (If 'all' is passed then all training data will be used)
        """
        shap_values = explainer.shap_values(samples)[1]

        vals = np.abs(shap_values).mean(0)
        feature_importance = pd.DataFrame(
            list(zip(samples.columns, vals)),
            columns=["col_name", "feature_importance_vals"],
        )
        feature_importance.sort_values(
            by=["feature_importance_vals"], ascending=False, inplace=True
        )

        feature_importance['rank'] = feature_importance['feature_importance_vals'].rank(method='max',ascending=False)

        missing_features = [
            i
            for i in columns_to_show
            if i not in feature_importance["col_name"][:20].tolist()
        ]
        missing_index = []
        for i in missing_features:
            missing_index.append(samples.columns.tolist().index(i))

        missing_features_new = []
        rename_col = {}
        for i in missing_features:
            rank = int(feature_importance[feature_importance['col_name']==i]['rank'].values)
            missing_features_new.append('rank:'+str(rank)+' - '+i)
            rename_col[i] = 'rank:'+str(rank)+' - '+i

        column_names = feature_importance["col_name"][:20].values.tolist() + missing_features_new

        feature_index = feature_importance.index[:20].tolist() + missing_index

        shap.summary_plot(
                shap_values[:, feature_index].reshape(
                    samples.shape[0], len(feature_index)
                ),
                    samples.rename(columns=rename_col)[column_names],
                    max_display=len(feature_index),
                )

        self.feature_importance = feature_importance
