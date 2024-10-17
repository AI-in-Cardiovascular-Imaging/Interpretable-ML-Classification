from sklearn.feature_selection import SelectKBest, RFE, SelectPercentile, SelectFromModel, f_classif, chi2
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import pandas as pd
from boruta import BorutaPy
import numpy as np

""" Different Feature Selection strategies and their parameters are defined. """
""" Two strategies are not used 'RFE' and 'Boruta' """
class FeatureSelector:
    def __init__(self, method, n_features): 
        self.method = method
        self.n_features = n_features
        self.selector = self.initialize_selector()
        
    def initialize_selector(self):
        if self.method == 'Select k Best':
            return SelectKBest(score_func=f_classif, k=self.n_features) 
        elif self.method == 'Rfe': 
            return RFE(estimator=RandomForestClassifier(n_estimators=100), n_features_to_select=self.n_features)       
        elif self.method == 'Select Percentile':    # Select features according to a percentile of the highest scores.
            return SelectPercentile(percentile=7)
        elif self.method == 'Select from Model': 
            return SelectFromModel(estimator=LinearSVC(), threshold='mean', max_features=self.n_features) # Meta-transformer for selecting features based on importance weights.
        elif self.method == 'Boruta':
            return BorutaPy(estimator=RandomForestClassifier(n_estimators=100), n_estimators='auto', max_iter=100)
        elif self.method == "Pca":
            return PCA(n_components=self.n_features)
        else:
            raise ValueError("Choose a valid feature selection method.")
    
    """implements .fit on data"""
    def fit(self, X, y=None):
        self.selector.fit(X, y) 
        return self

    """implements transform on data"""
    def transform(self, X):
         return self.selector.transform(X)
   
    """retures selected features by each feature selection method"""
    def get_selected_features(self, X):
        if self.method == 'Select k Best':
            return X.columns[self.selector.get_support(indices=True)].tolist() # Get the selected feature names for SelectKBest
        elif self.method == 'Rfe':
            return X.columns[self.selector.get_support(indices=True)].tolist() # Get the selected feature names for RFE
        elif self.method == 'Select Percentile':
            return X.columns[self.selector.get_support(indices=True)].tolist() # Get the selected feature names for SelectPercentile
        elif self.method == 'Select from Model':
            return X.columns[self.selector.get_support(indices=True)].tolist() # Get the selected feature names for SelectFromModel
        elif self.method == 'Boruta':
            return X.columns[self.selector.get_support(indices=True)].tolist()
        elif self.method == 'Pca':
            return self.get_important_pca_features(X)
        else:
             raise ValueError("Choose a valid feature selection method.")


    """PCA returns components, by this def, we get name and value of selected features"""
    def get_important_pca_features(self, X):
        """Get the loadings (coefficients) for each principal component"""
        loadings = self.selector.components_.T
        
        """Calculate the magnitude of the loadings"""
        loading_magnitudes = np.abs(loadings)
        feature_importance = np.sum(loading_magnitudes, axis=1)
        
        """Get the indices of the top features"""
        top_feature_indices = np.argsort(feature_importance)[-self.n_features:]
        
        """Get the names of the top features (columns headers)"""
        top_feature_names = X.columns[top_feature_indices].tolist()
        
        return top_feature_names

