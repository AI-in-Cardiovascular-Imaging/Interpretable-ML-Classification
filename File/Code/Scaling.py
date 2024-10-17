from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import pandas as pd

""" Different Scaling methods are defined"""
class Scaling:
    def __init__(self, method):
        self.method = method
        self.initialize_scaler()

    """based on scaling method, puts scaler value equal to Scaling library"""
    def initialize_scaler(self):
        if self.method == "Standard":
            self.scaler = StandardScaler()                 # Scale data between -1 and 1
        elif self.method == "MinMax":
            self.scaler = MinMaxScaler()                 # Scale data between 0 and 1
        elif self.method == 'RobustScaler':
            self.scaler = RobustScaler()
        else:
            raise ValueError("Invalid scaling method. Choose 'standard' or 'min_max'.")
        
        
    """fits selected scaler on data"""
    def fit(self, X, y=None):
        self.scaler.fit(X)
        return self

    """transforms fitted data"""
    def transform(self, X):
        if self.scaler is None:
            raise ValueError("Scaler not initialized. Call initialize_scaler() first.")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")

        return pd.DataFrame(self.scaler.transform(X), columns=X.columns, index=X.index)
