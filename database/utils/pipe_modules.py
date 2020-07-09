from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
import pandas as pd
import numpy as np


class CNAE_Transformer(BaseEstimator, TransformerMixin):

    # Return self nothing else to do here
    def fit(self, X, y=None):
        return self

    # Transformer method we wrote for this transformer
    def transform(self, X, y=None):
        X = X.copy()
        X.cnae = X.cnae.astype(str).str.strip()
        X.loc[:, "sector"] = X.cnae.astype(str).str[:2]

        X = X.replace({"sector": ""}, "missing")
        return X


class Mean_Imputer(BaseEstimator, TransformerMixin):

    # Return self nothing else to do here
    def fit(self, X, y=None):
        return self

    # Transformer method we wrote for this transformer
    def transform(self, X, y=None):
        numeric_column_names = X.select_dtypes(include=["float64", "int"]).columns
        X = X.copy()
        X[numeric_column_names] = X[numeric_column_names].fillna(X.mean())
        return X


class GroupNormalizer(BaseEstimator, TransformerMixin):
    '''
    Class used for imputing missing values in a pd.DataFrame using either mean or median of a group.

    Parameters
    ----------
    group_cols : list
        List of columns used for calculating the aggregated value
    target : str
        The name of the column to impute
    metric : str

    Returns
    -------
    X : array-like
        The array with imputed values in the target column
    '''

    def __init__(self, group_cols, target):

        self.group_cols = group_cols
        self.target = target

    def fit(self, X, y=None):

        assert pd.isnull(X[self.group_cols]).any(axis=None) == False, 'There are missing values in group_cols'

        impute_map = X.groupby(self.group_cols)[self.target].agg([np.mean, np.std]) \
            .reset_index(drop=False)
        self.impute_map_ = impute_map.fillna(impute_map.median())

        impute_map_total = X[self.target].agg([np.mean, np.std])
        self.impute_map_total = impute_map_total.fillna(impute_map_total.median())

        return self

    def normalizer_sector(self, df):

        df_normalized = pd.DataFrame(columns=df.columns)
        for group, x in df.groupby("sector"):
            if any(x.sector.isin(self.impute_map_.sector)):
                impute_sector = self.impute_map_.loc[self.impute_map_.sector.isin(x.sector)]

                mean = impute_sector.xs("mean", level=1, axis=1)
                std = impute_sector.xs("std", level=1, axis=1)
                x[self.target] = (x[self.target] - mean.iloc[0]) - (x[self.target] - std.iloc[0])
            else:
                x.loc[:, self.target] = (x[self.target] - self.impute_map_total.loc['mean']) / \
                                        self.impute_map_total.loc['std']
            df_normalized = df_normalized.append(x)
        return df_normalized

    def normalizer_total(self, df):

        df.loc[:, self.target] = (df[self.target] - self.impute_map_total.loc['mean']) / self.impute_map_total.loc[
            'std']
        return df

    def transform(self, X, y=None):

        # make sure that the imputer was fitted
        check_is_fitted(self, 'impute_map_')

        X = X.copy()
        df_final = pd.DataFrame(columns=X.columns)

        # Primero vemos si el sector de la tupla que queremos trasnformar está en el atributo impute_map_, que contiene
        # la media y la desviación estándar por grupo de cuando se entrenó el modelo. Si no está, aplicamos la normalización
        # basándonos en la media y la desviación estándar de todo el dataset con el que se entrenó.

        df_final = self.normalizer_sector(X)

        return df_final


# Custom Transformer that extracts columns passed as argument to its constructor
class FeatureSelector(BaseEstimator, TransformerMixin):
    # Class Constructor
    def __init__(self, feature_names):
        self._feature_names = feature_names

        # Return self nothing else to do here

    def fit(self, X, y=None):
        return self

        # Method that describes what we need this transformer to do

    def transform(self, X, y=None):
        return X[self._feature_names]


class MyLabelBinarizer(TransformerMixin):
    def __init__(self, *args, **kwargs):
        self.encoder = LabelBinarizer(*args, **kwargs)

    def fit(self, x, y=0):
        self.encoder.fit(x)
        return self

    def transform(self, x, y=0):
        return self.encoder.transform(x)

class MyLabelEncoder(TransformerMixin):
    def __init__(self, *args, **kwargs):
        self.encoder = LabelEncoder(*args, **kwargs)

    def fit(self, x, y=0):
        self.encoder.fit(x)
        return self

    def transform(self, x, y=0):
        return self.encoder.transform(x)