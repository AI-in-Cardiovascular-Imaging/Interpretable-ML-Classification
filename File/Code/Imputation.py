from sklearn.impute import SimpleImputer
import pandas as pd

"""Imputation method"""
class imputation:
    def __init__(self, strategy='Mean'):
        self.set_strategy(strategy)

    def set_strategy(self, strategy):
        valid_strategies = ["mean", "median", "most frequent"]
        if strategy not in valid_strategies:
            raise ValueError(f"Invalid strategy. Choose from {valid_strategies}")
        self.imputer = SimpleImputer(strategy=strategy)

    """fits selected imputation method on data"""
    def fit(self, X, y=None):
        self.imputer.fit(X, y)
        return self

    """After fitting, transforms imputed data"""
    def transform(self, X):
        if self.imputer is None:
            raise ValueError("Imputer not set. Use set_strategy() to define the imputation strategy.")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")

        return pd.DataFrame(self.imputer.transform(X), columns=X.columns, index=X.index)


