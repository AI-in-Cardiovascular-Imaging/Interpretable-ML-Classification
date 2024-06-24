from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, ShuffleSplit
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, roc_auc_score, accuracy_score
import json
import os
import numpy as np
import pandas as pd
import seaborn as sns
from FeatureSelector import FeatureSelector
import matplotlib.pyplot as plt
from Imputation import imputation
from Scaling import Scaling
from Classifiers import Classifiers
from Data_Reading import DataLoader, loading_main
from DataClean import DataCleaner, cleaner_main
from DataIdentifier import Identifier_main
import logging
import csv
import time
from tqdm import tqdm
import joblib
from sklearn.utils import shuffle
import pickle



file_path = f'/File/pkl_files/lr select k best min max best_model.pkl'
with open(file_path, 'rb') as file:
    data = joblib.load(file)

print(data)