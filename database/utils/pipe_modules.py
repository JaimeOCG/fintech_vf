from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
import pandas as pd
import numpy as np


class CNAE_Transformer(BaseEstimator, TransformerMixin):
    '''
    Class for replacing cnae code to sector group.
    Parameters
    ----------
    Returns
    -------
    X : DataFrame with new sector column
    '''
    # Return self nothing else to do here
    def fit(self, X, y=None):
        return self

    def conditions(self, df_rf):
        if df_rf['cnae'] < 510:
            return 'A'
        elif df_rf['cnae'] >= 510 and df_rf['cnae'] < 1011:
            return 'B'
        elif df_rf['cnae'] >= 1011 and df_rf['cnae'] < 3512:
            return 'C'
        elif df_rf['cnae'] >= 3512 and df_rf['cnae'] < 3600:
            return 'D'
        elif df_rf['cnae'] >= 3600 and df_rf['cnae'] < 4110:
            return 'E'
        elif df_rf['cnae'] >= 4110 and df_rf['cnae'] < 4511:
            return 'F'
        elif df_rf['cnae'] >= 4511 and df_rf['cnae'] < 4910:
            return 'G'
        elif df_rf['cnae'] >= 4910 and df_rf['cnae'] < 5510:
            return 'H'
        elif df_rf['cnae'] >= 5510 and df_rf['cnae'] < 5811:
            return 'I'
        elif df_rf['cnae'] >= 5811 and df_rf['cnae'] < 6411:
            return 'J'
        elif df_rf['cnae'] >= 6411 and df_rf['cnae'] < 6810:
            return 'K'
        elif df_rf['cnae'] >= 6810 and df_rf['cnae'] < 6910:
            return 'L'
        elif df_rf['cnae'] >= 6910 and df_rf['cnae'] < 7711:
            return 'M'
        elif df_rf['cnae'] >= 7711 and df_rf['cnae'] < 8411:
            return 'N'
        elif df_rf['cnae'] >= 8411 and df_rf['cnae'] < 8510:
            return 'O'
        elif df_rf['cnae'] >= 8510 and df_rf['cnae'] < 8610:
            return 'P'
        elif df_rf['cnae'] >= 8610 and df_rf['cnae'] < 9001:
            return 'Q'
        elif df_rf['cnae'] >= 9001 and df_rf['cnae'] < 9411:
            return 'R'
        elif df_rf['cnae'] >= 9411 and df_rf['cnae'] < 9700:
            return 'S'
        elif df_rf['cnae'] >= 9700 and df_rf['cnae'] < 9900:
            return 'T'
        elif df_rf['cnae'] >= 9900:
            return 'S'
        else:
            return 'Unknown'

    # Transformer method we wrote for this transformer
    def transform(self, X, y=None):
        X = X.copy()
        X = X.dropna(subset=['cnae'])
        X.cnae = X.cnae.astype(int)
        X.loc[:, "sector"] = X.apply(self.conditions, axis=1)

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
    X : pd.DataFrame with normalized columns by sector
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
                #x.loc[:, self.target] = (x[self.target] - self.impute_map_total.loc['mean'])
                x[self.target] = (x[self.target] - mean.iloc[0]).div(std.iloc[0])
            else:
                x.loc[:, self.target] = (x[self.target] - self.impute_map_total.loc['mean']).div(self.impute_map_total.loc['std'])
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