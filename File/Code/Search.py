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
from Plotting import Plotting
import logging
import csv
import time
import shap
from tqdm import tqdm
import joblib
from joblib import dump
from threading import current_thread


"""To measure the run time of search algorithm"""
def measure_runtime(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result_time = func(*args, **kwargs)
        end_time = time.time()
        runtime = end_time - start_time
        print(f"Runtime for {func.__name__}: {runtime} seconds")
        return result_time
    return wrapper


class SearchMethod:

    def __init__(self, file_path, method='GridSearch'):
        self.method = method
        self.results_dict = {}           
        self.test_results_dict = {}      
        self.train_TEST_dict = {}       
        self.test_TEST_dict = {}        
        self.results_train_list = []
        self.results_test_list = []
        self.test_results_csv_list = []
        self.train_results_csv_list = []
        self.run_times = [] 
        self.roc_train_all = {}                        
        self.roc_test_all = {}                        
        self.precision_recall_train_all = {}                
        self.precision_recall_test_all = {}                
        self.prepared_data = pd.read_csv(file_path)
        self.classifier_obj = Classifiers()
        self.split_data()
        # self.preprocess_data()
        self.search_algorithm()


    """Splits data to train and test"""
    def split_data(self):
        X = self.prepared_data.drop(columns=["class"])
        y = self.prepared_data["class"]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)       # Data stratified on label "class"

    
    """ Main search function. this function consists of nested for loops to run on different combination of classfiers, feature selection, scaling method and imputation method"""
    @measure_runtime
    def search_algorithm(self):
        feature_selection_algorithms = ['Select Percentile', 'Select from Model', 'Select k Best', 'Pca', 'Rfe'] 
        scaling_methods = ['Standard', 'RobustScaler', 'MinMax']

        """NEW to check if there is thread problem in run time"""
        if not current_thread().name == 'MainThread': 
            raise RuntimeError('Search_algorithm must be run in the main thread')
        
        """changed the place of feature_selection and clf_name loop. scaling loop came at first instead of feature selection"""
        for scaling_method in scaling_methods:
            for feature_selection in feature_selection_algorithms:
                for clf_name, clf in self.classifier_obj.classifiers.items():
                    start_time = time.time()           
                    pipeline = main_pipeline.get_pipeline(scaling_method, feature_selection, clf)  
                    print(f"Pipeline steps: {pipeline.steps}")      
                    cv = ShuffleSplit(n_splits=10, random_state=2)
                    if self.method == "GridSearch":
                        search_algorithm = GridSearchCV(pipeline, param_grid=self.classifier_obj.param_grids[clf_name], cv=cv, n_jobs=1)  
                    elif self.method == "RandomSearch":
                        search_algorithm = RandomizedSearchCV(pipeline, param_distributions=self.classifier_obj.param_grids[clf_name], cv=cv, n_jobs=1) 
                    else:
                        raise ValueError("Unknown search algorithm")
                    search_algorithm.fit(self.X_train, self.y_train)

                    end_time = time.time()
                    run_time = end_time - start_time  
                    self.run_times.append(run_time)
                    
                    output_folder = f'/File/Results/{clf}'      
                    os.makedirs(output_folder, exist_ok=True)
                    self.evaluate_on_train(search_algorithm, feature_selection, output_folder, clf_name, scaling_method)

                    self.evaluate_on_test(search_algorithm, feature_selection, output_folder, clf_name, scaling_method)
                    self.save_result_to_json(clf_name, output_folder, feature_selection)

                    """save .pkl file of each classifier"""
                    pkl_path = f'/File/pkl_files' 

                    """save the result of trained models with parameters"""
                    self.best_estimator = search_algorithm.best_estimator_
                    dump(self.best_estimator, os.path.join(pkl_path, f'{clf_name} {feature_selection} {scaling_method} best_model.pkl'))


    """Search algorithm, after training in search function, runs on train part of data to see what is the output """
    def evaluate_on_train(self, search_algorithm, feature_selection, output_folder, clf_name, scaling_method):   
        clf = f'{clf_name} {feature_selection} {scaling_method}'     
        start_train_time = time.time()
        y_pred_proba_train = search_algorithm.predict_proba(self.X_train)[:, 1]
        fpr_train, tpr_train, thresholds_fpr_tpr_train = roc_curve(self.y_train, y_pred_proba_train)
        auc_roc_train = roc_auc_score(self.y_train, y_pred_proba_train)
        precision_train, recall_train, thresholds_pr_train = precision_recall_curve(self.y_train, y_pred_proba_train)
        f1 = 2 * (precision_train * recall_train) / (precision_train + recall_train)

        y_pred_train = search_algorithm.predict(self.X_train)           
        cm_train = confusion_matrix(self.y_train, y_pred_train)
        train_accuracy = accuracy_score(self.y_train, y_pred_train)
        best_params_train = search_algorithm.best_params_
        TN, FP, FN, TP = cm_train.ravel()     
        run_time = self.run_times[-1]         
        imputation_strategy = 'mean'
        scaling_method = scaling_method

        """Calculating Youden Index"""
        youden_index_train = tpr_train - fpr_train

        """Finding the threshold which maximizes youden index"""
        optimal_thresh_index_train = np.argmax(youden_index_train)
        optimal_thresh_train = thresholds_fpr_tpr_train[optimal_thresh_index_train]

        """Calculating Specificity and Sensitivity by using optimal threshold"""
        sensitivity_train = tpr_train[optimal_thresh_index_train]
        specificity_train = 1 - fpr_train[optimal_thresh_index_train]

        """Calculating the ACCURACy based on youden output"""
        y_pred_train_youden = (y_pred_proba_train >= optimal_thresh_train).astype(int)
        accuracy_train_youden = accuracy_score(self.y_train, y_pred_train_youden)

        """Calculating AUC based on optimized threshold by youden index"""
        auc_youden = roc_auc_score(self.y_train, y_pred_proba_train >= optimal_thresh_train)

        """Save the Selected Features by values before normalization in a .csv file"""
        selected_features_train = search_algorithm.best_estimator_.named_steps['feature_selection'].get_selected_features(self.X_train)   
        selected_features_values_train = self.X_train[selected_features_train]            
        selected_features_by_values_train = pd.concat([pd.DataFrame(selected_features_values_train), pd.DataFrame(self.y_train)], axis=1)
        selected_features_by_values_train.to_csv(os.path.join(output_folder, f"{clf_name} {feature_selection} selected features train.csv"), index=False)   
    
        try:
            if hasattr(search_algorithm.best_estimator_.named_steps['classifier'], 'feature_importances_'):
                importance_scores = search_algorithm.best_estimator_.named_steps['classifier'].feature_importances_ 
                top_k_indices = importance_scores.argsort()[-100:][::-1]
                top_k_features = [selected_features_train[i] for i in top_k_indices]
                text_name = f"{clf_name}_important_features_indices_train"
                with open(os.path.join(output_folder, f"{text_name}.txt"), "w") as file:
                    for feature in top_k_features:
                        file.write(feature + '\n')
            else:
                print(f"{clf} does not support feature importances!!!!")

        except AttributeError as e:
            print(f"Error accessing feature importances for {clf}: {e}")
        
        end_train_time = time.time()
        train_time = end_train_time - start_train_time

        train_result = {
            "Imputation Algorithm": imputation_strategy,            
            "Scaling Method": scaling_method,                
            "Feature Selection": feature_selection, 
            "Classifier": clf_name,
            "Run Time(s) Train": train_time,      
            "Accuracy Train": train_accuracy,
            "Best parameters Train": best_params_train,
            "ROC-AUC Train": auc_roc_train.tolist(),
            "Precision Values Train": precision_train.tolist(),    
            "Precision Train": np.mean(precision_train),
            "Recall Values Train": recall_train.tolist(),          
            "Recall Train": np.mean(recall_train),
            "F1-score Values Train": f1.tolist(),     
            "F1-score Train": np.mean(f1),
            "Confusion_matrix Train": cm_train.tolist(),
            "Thresholds FPR_TPR Train": thresholds_fpr_tpr_train.tolist(),     
            "Thresholds Precision Recall Train": thresholds_pr_train.tolist(),
            "True Positive Train": TP,
            "True Negative Train": TN,
            "False Positive Train": FP,
            "False Negative Train": FN,
            "Youden Index Train": youden_index_train[optimal_thresh_index_train], 
            "Optimal Threshold Train": optimal_thresh_train,                
            "Sensitivity Train": sensitivity_train,                        
            "Specificity Train": specificity_train,                        
            "ROC-AUC (youden index) Train": auc_youden,
            "Accuracy (youden) Train": accuracy_train_youden,
            "Main Run Time(s)":run_time,
        }

        self.results_train_list.append(train_result)
        keys_to_exclude = ["True Positive Train", "True Negative Train", "False Positive Train", "False Negative Train", "Best parameters Train"]
        self.train_TEST_dict = {key: value for key, value in train_result.items() if key not in keys_to_exclude}
        self.train_results_dict = {k: v for k, v in train_result.items() if k not in ['Thresholds Precision Recall Train',  
                                                                                "Thresholds FPR_TPR Train", "Confusion_matrix Train", "Precision Values Train",
                                                                                 "Recall Values Train", "F1-score Values Train"]}
        self.train_results_csv_list.append(self.train_results_dict)
        self.roc_train_all[clf] = (fpr_train, tpr_train, auc_roc_train)                           
        self.precision_recall_train_all[clf] = (precision_train, recall_train)      

        """Different plots of output values"""
        plotting = Plotting()
        plotting.roc_curve_combined(clf_name, self.roc_train_all, output_folder, subset='train')       
        plotting.precision_recall_combined(clf_name, self.precision_recall_train_all, output_folder, subset='train')
        plotting.plot_confusion_matrix(cm_train, clf, output_folder, subset='train')                
        plotting.plot_roc_curve(fpr_train, tpr_train, clf, output_folder, optimal_thresh_index_train, subset='train')
        plotting.plot_precision_recall(precision_train, recall_train, clf, output_folder, subset='train')


    """print the results in output"""
    def display_results_train(self):
        for result in self.results_train_list:
            print(f"Results for {result['Classifier']}_train:")
            print(f"Imputation: {result['Imputation Algorithm']}")               
            print(f"Scaling: {result['Scaling Method']}")                       
            print(f"Feature Selection: {result['Feature Selection']}")
            print(f"Train Accuracy: {result['Accuracy Train']}")
            print(f"Best Parameters: {result['Best parameters Train']}")
            print(f"ROC-AUC: {result['ROC-AUC Train']}")
            print(f"Precision: {np.mean(result['Precision Train'])}")
            print(f"Recall: {np.mean(result['Recall Train'])}")
            print(f"F1-score: {np.mean(result['F1-score Train'])}")
            print(f"Confution matrix: {result['Confusion_matrix Train']}")
            print(f"Value of metrics: {result['True Positive Train']} and {result['True Negative Train']} and {result['False Positive Train']} and {result['False Negative Train']}")
            print(f"Youden Index: {result['Youden Index Train']}")
            print(f"Optimal Threshold: {result['Optimal Threshold Train']}")
            print(f"Sensitivity: {result['Sensitivity Train']}")
            print(f"Specificity: {result['Specificity Train']}")
            print(f"ROC-AUC youden index Train: {result['ROC-AUC (youden index) Train']}")
            print(f"Accuracy (youden) Train: {result['Accuracy (youden) Train']}")
            print(f"Run time: {result['Main Run Time(s)']}")
            print(f"Run Time Train: {result['Run Time(s) Train']}")
            print("\n")
        
    
    """Result of search algorithms on test set"""
    def evaluate_on_test(self, search_algorithm, feature_selection, output_folder, clf_name, scaling_method):   
        clf = f'{clf_name} {feature_selection} {scaling_method}'          
        start_test_time = time.time()
        y_pred_proba_test = search_algorithm.predict_proba(self.X_test)[:, 1]
        fpr_test, tpr_test, thresholds_fpr_tpr_test = roc_curve(self.y_test, y_pred_proba_test)
        auc_roc_test = roc_auc_score(self.y_test, y_pred_proba_test)
        precision_test, recall_test, thresholds_pr_test = precision_recall_curve(self.y_test, y_pred_proba_test)
        f1_test = 2 * (precision_test * recall_test) / (precision_test + recall_test)

        y_pred_test = search_algorithm.predict(self.X_test)
        cm_test = confusion_matrix(self.y_test, y_pred_test)
        best_params_test = search_algorithm.best_params_
        test_accuracy = accuracy_score(self.y_test, y_pred_test)
        TN, FP, FN, TP = cm_test.ravel()  
        # run_time_test = self.run_times[-1]                  
        imputation_strategy = 'mean'
        scaling_method = scaling_method

        """Calculating Youden Index"""
        youden_index_test = tpr_test - fpr_test

        """Finding the threshold which maximizes youden index"""
        optimal_thresh_index_test = np.argmax(youden_index_test)
        optimal_thresh_test = thresholds_fpr_tpr_test[optimal_thresh_index_test]

        """Calculating Specificity and Sensitivity by using optimal threshold"""
        sensitivity_test = tpr_test[optimal_thresh_index_test]
        specificity_test = 1 - fpr_test[optimal_thresh_index_test]

        """Calculating the ACCURACy based on youden output"""
        y_pred_test_youden = (y_pred_proba_test >= optimal_thresh_test).astype(int)
        accuracy_test_youden = accuracy_score(self.y_test, y_pred_test_youden)

        """AUC based on youden index"""
        auc_youden_test = roc_auc_score(self.y_test, y_pred_proba_test > optimal_thresh_test)


        """Save the Selected Features by values before normalization in a .csv file"""
        selected_features_test = search_algorithm.best_estimator_.named_steps['feature_selection'].get_selected_features(self.X_test)   
        selected_features_values_test = self.X_test[selected_features_test]                
        selected_features_by_values_test = pd.concat([pd.DataFrame(selected_features_values_test), pd.DataFrame(self.y_test)], axis=1 )   
        selected_features_by_values_test.to_csv(os.path.join(output_folder, f"{clf_name} {feature_selection} selected features test.csv"), index=False)   

        try:
            if hasattr(search_algorithm.best_estimator_.named_steps['classifier'], 'feature_importances_'):
                importance_scores = search_algorithm.best_estimator_.named_steps['classifier'].feature_importances_
                top_k_indices = importance_scores.argsort()[-100:][::-1]
                top_k_features = [selected_features_test[i] for i in top_k_indices]
                text_name = f"{clf_name}_important_features_indices_test"
                with open(os.path.join(output_folder, f"{text_name}.txt"), "w") as file:
                    for feature in top_k_features:
                        file.write(feature + '\n')
            else:
                print(f"{clf} does not support feature importances!!!!")

        except AttributeError as e:
            print(f"Error accessing feature importances for {clf}: {e}")
        
        end_test_time = time.time()
        test_time = end_test_time - start_test_time

        test_result = {
            "Imputation Algorithm": imputation_strategy,             
            "Scaling Method": scaling_method,                 
            "Feature Selection": feature_selection,
            "Classifier": clf_name,
            "Run Time Test(s)": test_time,
            "Accuracy Test": test_accuracy,
            "Best parameters Test": best_params_test,
            "ROC-AUC Test": auc_roc_test,
            "Precision Values Test": precision_test.tolist(),
            "Precision Test": np.mean(precision_test),
            "Recall Values Test": recall_test.tolist(),
            "Recall Test": np.mean(recall_test),
            "F1-score Values Test": f1_test.tolist(),   
            "F1-score Test":np.mean(f1_test),
            "Thresholds FPR_TPR Test": thresholds_fpr_tpr_test.tolist(),
            "Thresholds Precision_Recall Test": thresholds_pr_test.tolist(),
            "Confusion_matrix Test": cm_test.tolist(),
            "True Positive Test": TP,
            "True Negative Test": TN,
            "False Positive Test": FP,
            "False Negative Test": FN,
            "Youden Index Test": youden_index_test[optimal_thresh_index_test], 
            "Optimal Threshold Test": optimal_thresh_test,                
            "Sensitivity Test": sensitivity_test,                        
            "Specificity Test": specificity_test, 
            "Accuracy (youden) Test": accuracy_test_youden,                        
            "ROC-AUC (youden index) Test": auc_youden_test,
            
        }

        self.results_test_list.append(test_result)
        keys_to_exclude = ["True Positive Test", "True Negative Test", "False Positive Test", "False Negative Test", "Best parameters Test"]
        self.test_TEST_dict = {key: value for key, value in test_result.items() if key not in keys_to_exclude}
        self.test_results = {k: v for k, v in test_result.items() if k not in ["Thresholds FPR_TPR Test", "Thresholds Precision_Recall Test", 
                                                                               "Confusion_matrix Test", "Precision Values Test", "Recall Values Test",
                                                                               "F1-score Values Test"]}
        self.test_results_csv_list.append(self.test_results)
        self.roc_test_all[clf] = (fpr_test, tpr_test, auc_roc_test)                                  
        self.precision_recall_test_all[clf] = (precision_test, recall_test)      
    
        plotting = Plotting()
        plotting.roc_curve_combined(clf_name, self.roc_test_all, output_folder, subset='test') 
        plotting.precision_recall_combined(clf_name, self.precision_recall_test_all, output_folder, subset='test')
        plotting.plot_confusion_matrix(cm_test, clf, output_folder, subset='test')                   
        plotting.plot_roc_curve(fpr_test, tpr_test, clf, output_folder, optimal_thresh_index_test, subset='test')
        plotting.plot_precision_recall(precision_test, recall_test, clf, output_folder, subset='test')


    """ prints the result of test set"""
    def display_results_test(self):
        for result in self.results_test_list:
            print(f"Results for {result['Classifier']}_test:")
            print(f"Imputation: {result['Imputation Algorithm']}")                
            print(f"Scaling: {result['Scaling Method']}")                       
            print(f"Feature Selection: {result['Feature Selection']}")  
            print(f"Best test Score: {result['Accuracy Test']}")
            print(f"Best Parameters: {result['Best parameters Test']}")
            print(f"ROC-AUC Test: {result['ROC-AUC Test']}")
            print(f"Precision Test: {np.mean(result['Precision Test'])}")
            print(f"Recall Test: {np.mean(result['Recall Test'])}")
            print(f"F1-score Test: {np.mean(result['F1-score Test'])}")
            print(f"Confution matrix: {result['Confusion_matrix Test']}")
            print(f"Value of metrics: {result['True Positive Test']} and {result['True Negative Test']} and {result['False Positive Test']} and {result['False Negative Test']}")
            print(f"Youden Index: {result['Youden Index Test']}")
            print(f"Optimal Threshold: {result['Optimal Threshold Test']}")
            print(f"Sensitivity: {result['Sensitivity Test']}")
            print(f"Specificity: {result['Specificity Test']}")
            print(f"Accuracy (youden) Test: {result['Accuracy (youden) Test']}")
            print(f"ROC-AUC youden index Test: {result['ROC-AUC (youden index) Test']}")
            print(f"Run time: {result['Run Time Test(s)']}")            
            print("\n")
        
    """ Save the result of each combination in a .json file"""
    def save_result_to_json(self,clf_name, output_folder, feature_selection):
        filename = os.path.join(output_folder, f"{clf_name}_{feature_selection}_results.json")
        combined_results = {'training_results': self.train_TEST_dict, 'testing_results': self.test_TEST_dict}
        with open(filename, "w") as json_file:
            json.dump(combined_results, json_file, indent=4)

    """Save the final results in a .csv file and plot outputs in different heatmap plots"""
    def merge_and_save_to_csv(self):        # clf_name removed
        train_df = pd.DataFrame(self.train_results_csv_list)
        test_df = pd.DataFrame(self.test_results_csv_list)
        
        """Merge DataFrames on common keys"""
        common_keys = ["Imputation Algorithm", "Scaling Method", "Feature Selection", "Classifier"]
        combined_df = pd.merge(train_df, test_df, on=common_keys, suffixes=('_train', '_test'))
        
        """Remove duplicate rows based on common keys"""
        combined_df.drop_duplicates(subset=common_keys, inplace=True)

        output_folder = f'/File/Results'            
        filename = f"Final_Results.csv"
        filepath = os.path.join(output_folder, filename)
        combined_df.to_csv(filepath, index=False)


        # Acc = ['Accuracy Test']
        # Plotting.heatmap_of_results_of_metric(combined_df, Acc, output_folder)

        # Spe = ['Specificity Test']
        # Plotting.heatmap_of_results_of_metric(combined_df, Spe, output_folder)

        # Sen = ['Sensitivity Test']
        # Plotting.heatmap_of_results_of_metric(combined_df, Sen, output_folder)

        # AUC = ['ROC-AUC Test']
        # Plotting.heatmap_of_results_of_metric(combined_df, AUC, output_folder)

        # metrics = ['Accuracy Test', 'Sensitivity Test', 'Specificity Test', 'ROC-AUC Test']
        # Plotting.heatmap_of_results(combined_df, metrics, output_folder)

        SEN_test = ['Sensitivity Test']
        Plotting.heatmap_of_results(combined_df, SEN_test, output_folder)

        ACC_test = ['Accuracy Test']
        Plotting.heatmap_of_results(combined_df, ACC_test, output_folder)

        SPE_test = ['Specificity Test']
        Plotting.heatmap_of_results(combined_df, SPE_test, output_folder)

        ROC_AUC_test = ['ROC-AUC Test']
        Plotting.heatmap_of_results(combined_df, ROC_AUC_test, output_folder)
        
    

""" A pipeline to have different classifiers and feature selection and imputatation and scaling algorithms"""
class main_pipeline:
    def get_pipeline(scaling_method, feature_selection, classifier):       
        pipeline = Pipeline(steps=[
            ('imputation_method', imputation(strategy='mean')),
            ('scaling_method', Scaling(method=scaling_method)),
            ('feature_selection', FeatureSelector(method=feature_selection, n_features=35)),
            ("classifier", classifier)
        ])
        return pipeline

def main_search():
    logging.basicConfig(level=logging.INFO)
    file_path = f'/File/Results/final_file.csv'
    search = SearchMethod(file_path)
    search.display_results_train()
    search.display_results_test()
    search.merge_and_save_to_csv()          # 'classifier' removed

