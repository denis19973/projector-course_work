import catboost as ctb
import pandas as pd


MODEL_PATH = '../predictor/models/catboost_model_with_pool'


def predict(X):
    model = ctb.CatBoostRegressor()
    if 'Unnamed: 0' in X.columns:
        X = X.drop('Unnamed: 0', axis=1)
    model.load_model(MODEL_PATH)
    return pd.Series(model.predict(X))
