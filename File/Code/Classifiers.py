from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier, GradientBoostingClassifier, StackingClassifier
import numpy as np
import xgboost as xgb




class Classifiers:
    def __init__(self):
        self.classifiers = self.initialize_classifiers()
        self.param_grids = self.initialize_param_grids()

    def initialize_classifiers(self):
        return {
            "Lr": LogisticRegression(max_iter=10000, tol=0.1),
            "Dt": DecisionTreeClassifier(),
            "Knn": KNeighborsClassifier(),
            "Svc": SVC(probability=True),
            "Rf": RandomForestClassifier(max_depth=4),
            "Ada boost": AdaBoostClassifier(),
            "Extra tree": ExtraTreesClassifier(),
            'Bagging':BaggingClassifier(),
            'Gboost': GradientBoostingClassifier(),                                        
            # 'Xgboost': xgb.XGBClassifier(),      
            # 'Stacking':StackingClassifier(estimators=[('rf', RandomForestClassifier(n_estimators=15))]), 
        }   

    def initialize_param_grids(self):
        n_estimators = [int(x) for x in np.linspace(start=10, stop=500, num=10)]
        max_features = ['auto', 'sqrt', 'log2']
        max_features_extra = ['sqrt', 'log2']
        max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
        learning_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
        return {
            "Lr": {"classifier__C": np.logspace(-3, 3, 10)},
            "Dt": {"classifier__max_depth":np.arange(2, 30, 2), "classifier__max_features":max_features, "classifier__criterion":['gini', 'entropy']},
            "Knn": {"classifier__n_neighbors": np.arange(1, 25, 2)},
            "Svc": {"classifier__C": np.logspace(-4, 4, 8), "classifier__gamma":np.logspace(-4, 4, 8)},
            "Rf": {"classifier__n_estimators": n_estimators, "classifier__max_features": max_features,
                   "classifier__min_samples_split": [2, 5, 10], "classifier__min_samples_leaf": [1, 3, 4],
                   "classifier__random_state": [42]},
            
            "Ada boost":{"classifier__n_estimators": [(20*i) for i in range(1, 6)], "classifier__learning_rate": [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5]},
            "Extra tree": {"classifier__n_estimators": [int(x) for x in np.linspace(start=10, stop=1000, num=10)], "classifier__max_features": max_features_extra},
            "Bagging":{"classifier__n_estimators":[int(x) for x in np.linspace(start=2, stop=500, num=15 )]},
            "Gboost": {"classifier__n_estimators": n_estimators, "classifier__learning_rate": learning_rate, "classifier__max_depth": max_depth, "classifier__subsample": [0.8, 0.9, 1.0]},   
            # "Xgboost":{},          # NEW added
            # "Stacking":{},
        }
