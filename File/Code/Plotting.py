import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from matplotlib.colors import LinearSegmentedColormap


"""Different plots and figures of metrics like ROC, AUC, Confusion Matrix"""
class Plotting:

    """AUC-ROC curve of each combination"""
    def plot_roc_curve(self, fpr, tpr, clf, output_folder, optimal_thresh_index, subset='train'):
        self.optimal_threshold_index = optimal_thresh_index
        plt.figure(figsize=(14, 12))
        plt.plot(fpr, tpr, color='orange', label = 'ROC')
        plt.plot([0, 1], [0, 1], color='darkblue', linestyle = '--')
        plt.scatter(fpr[self.optimal_threshold_index], tpr[self.optimal_threshold_index], color='red', marker='o', s=100, alpha=0.5, label='Optimal Threshold') # show optimal threshold point on plot
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_folder, f"{clf} Roc curve {subset}.png"), dpi = 300)  
        plt.close()     

    """Precission-Recall plot of each combination"""
    def plot_precision_recall(self, precision, recall, clf, output_folder, subset='train'):
        plt.figure(figsize=(14, 12))
        plt.plot(precision, recall, color = "blue", lw = 2, label = 'Precision-Recall') 
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Recal")
        plt.ylabel("Precision")
        plt.title("Precision-Recall{}".format(clf))
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.savefig(os.path.join(output_folder, f"{clf} precision recall {subset}.png"), dpi=300) 
        plt.close()    

    """Confustion matrix of result of each combination"""
    def plot_confusion_matrix(self, cm, clf, output_folder, subset='train'):        # clf_name removed and clf added, feature_selection removed
        plt.figure(figsize=(12, 9))

        """Calculate percentages of each output"""    
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        labels = [f'{value}\n{percent:.2%}' for value, percent in zip(cm.flatten(), cm_percent.flatten())]
        labels = np.array(labels).reshape(cm.shape[0], cm.shape[1])
        
        sns.heatmap(cm, annot=labels, fmt='', cmap='Oranges', xticklabels=["normal", "abnormal"], yticklabels=["normal", "abnormal"], 
                    linewidths=1, linecolor='white', center=.5,
                    annot_kws={"fontsize": 13, 'fontstyle': 'normal', 'color': 'black', "weight": "bold", "family": "serif"}) 
               
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.tight_layout() 
        # plt.savefig(os.path.join(output_folder, f'{clf_name} {feature_selection} confusion matrix {subset}.png'))     Original code
        plt.savefig(os.path.join(output_folder, f"{clf} confusion matrix {subset}.png"), dpi=300)  
        plt.close()  

    """Heatmap of all metrics defined in a list based on combination of classifier, scaling method and feature selection"""
    def heatmap_of_results(data, metric, output_folder):

        data['combination'] = data['Classifier'] + ' \ ' + data['Scaling Method']
        data.set_index('combination', inplace = True)
        
        pivot_data = data.pivot_table(index='combination', columns='Feature Selection', values=metric)
        if isinstance(pivot_data.columns, pd.MultiIndex):
            pivot_data.columns = pivot_data.columns.get_level_values(1)
                
        plt.figure(figsize=(14, 12), dpi=220)
        sns.heatmap(pivot_data, annot=True, cmap='Oranges', cbar=True, center=.5,
                    linewidths=.5, linecolor='white', fmt=".2g")
        plt.title(f'Heatmap of {metric}')
        plt.xlabel('Feature Selector')
        plt.ylabel('Classifiers')
        plt.xticks(rotation=30, fontsize=10)
        plt.yticks(rotation=15, fontsize=10)
        
        plt.savefig(os.path.join(output_folder, f"All Results of {metric}.png"))
        plt.close()


    """Heatmap of a metric like Accuracy"""
    def heatmap_of_results_of_metric(data, metric, output_folder):

        pivot_data = data.pivot_table(index = 'Classifier', columns='Feature Selection', values = metric)
        
        if isinstance(pivot_data.columns, pd.MultiIndex):
            pivot_data.columns = pivot_data.columns.get_level_values(1)
        
        plt.figure(figsize=(12, 8), dpi=250)
        sns.heatmap(pivot_data, annot=True, cmap= 'Wistia', cbar=True, center=.5, linecolor='white',
                    linewidths=.5, fmt=".2g", annot_kws={"fontsize": 13, 'fontstyle': 'normal', 'color': 'black', "weight": "bold", "family": "serif"})
        plt.title(f'{metric}', fontsize=20)
        plt.xlabel('Feature Selector', fontsize=15)
        plt.ylabel('Classifier', fontsize=15)
        plt.xticks(rotation = 10, fontsize = 13)
        plt.yticks(rotation=0, fontsize = 13)
        plt.savefig(os.path.join(output_folder, f"Heatmap of {metric}"))
        plt.close()
    

    """Combind plot of all AUC-ROC curves"""
    def roc_curve_combined(self, clf_name, roc_data, output_folder, subset='train'):         
        plt.figure(figsize=(12, 10), dpi=250)

        for clf, (fpr, tpr, auc_score) in roc_data.items():
            if clf_name in clf:                                  
                plt.plot(fpr, tpr, lw=1, label=f'{clf} (AUC = {auc_score:.2f})')     
        plt.plot([0, 1], [0, 1], color = 'navy', lw=2, linestyle='--')       
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title(f'ROC')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.savefig(os.path.join(output_folder, f'{clf_name} Combined AUC-ROC curve {subset}.png'))   
        plt.close()


    """Combined plot of all Precision-Recall curves"""
    def precision_recall_combined(self, clf_name, precision_recall_data, output_folder, subset='train'):
        plt.figure(figsize=(14, 12))

        for clf, (precision, recall) in precision_recall_data.items():
            if clf_name in clf:
                plt.plot(precision, recall, lw=2, label=f'{clf}')
        
        plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Recal")
        plt.ylabel("Precision")
        plt.title("Precision-Recall")
        plt.legend(loc='upper left')
        plt.grid(False)
        plt.savefig(os.path.join(output_folder, f'{clf_name} Precision-Recall {subset}.png'), dpi=300) 
        plt.close()    



    def correlation_plot(self, data, clf, clf_name, output_folder, subset = 'train'):
        cmap_dict = {
            0: '#e0f7fa', 1: '#b2ebf2', 2: '#80deea', 3: '#4dd0e1', 4: '#26c6da', 
            5: '#00acc1', 6: '#0097a7', 7: '#00838f', 8: '#006064', 9: '#004d40'   
        }

        cmap = ListedColormap([cmap_dict[i] for i in range(10)])
        # df = pd.DataFrame(data)
        corr_df = data.corr()
        plt.figure(figsize=(18, 16), dpi=300)
        sns.heatmap(corr_df, annot=True, cmap=cmap, vmin=-1, vmax=1, linewidths=0.5, square=True,
                    annot_kws={"fontsize": 13, 'fontstyle': 'normal', 'color': 'black', "weight": "bold", "family": "serif"})
        plt.title(f'{clf} correlation plot {subset}')
        plt.tight_layout()
        plt.xticks(rotation = 90)
        plt.yticks(rotation=0)
        plt.savefig(os.path.join(output_folder, f"{clf_name} {clf} Correlation plot {subset}.png"))
        plt.close() 


    def bitmap(self, data, clf,  clf_name, output_folder, subset='train'):

        data_subset = data.iloc[:, 0:15]
        if data_subset.shape[0] != data_subset.shape[1]:
            print("Warning: The subset is not square. Adjusting to the largest possible square subset.")
            min_dim = min(data_subset.shape)
            data_subset = data_subset.iloc[:min_dim, :min_dim]
        plt.figure(figsize=(24, 20), dpi=300)
        # data = data.iloc[:, 0:6]
        sns.heatmap(data_subset, cmap='Blues', cbar=True, annot=True, linewidths=0.45, linecolor='black', center=True,
                    annot_kws={"fontsize": 13, 'fontstyle': 'normal', 'color': 'black', "weight": "bold", "family": "serif"})
        plt.title(f'Bitmap Plot')
        plt.xticks(rotation = 90)
        plt.yticks(rotation=0)
        plt.savefig(os.path.join(output_folder, f"{clf_name} {clf} Bitmap plot {subset}.png"))
        plt.close()



    def box_plot(self, data, clf, clf_name, output_folder,subset = 'train'):
        plt.figure(figsize=(18, 16))
        df = data
        sns.boxplot(data=df)
        plt.title(f'Box plot')
        plt.tight_layout()
        plt.xticks(rotation = 90)
        plt.yticks(rotation=0)
        plt.savefig(os.path.join(output_folder, f"{clf_name} {clf} Box plot_{subset}.png"))
        plt.close()

    def cat_plot(self, data, clf, clf_name, output_folder, subset = 'train'):
        plt.figure(figsize=(18, 16))
        # df = pd.DataFrame(data)
        x = data.drop(columns = ['class'])
        y = data['class']
        sns.catplot(data=x, kind='box', height=8, aspect='auto')     
        plt.title(f'Categorical Plot')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.xticks(rotation = 90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"{clf_name} {clf} catplot {subset}.png"))
        plt.close()
    

    """Kernel Density Estimation plot"""
    def kdeplot(sefl, data, clf, clf_name, output_folder, subset='train'):
        plt.figure(figsize=(18, 16))
        x = data.drop(columns=['class'])
        y = data['class']
        sns.kdeplot(data=x, common_norm=False, cbar=True)
        plt.title(f'kde plot')
        plt.tight_layout()
        plt.xticks(rotation = 90)
        plt.yticks(rotation=0)
        plt.savefig(os.path.join(output_folder, f"{clf_name} {clf} kdeplot {subset}.png"))
        plt.close()


    """Histogram plot """
    def histoplot(self, data, clf, clf_name, output_folder, subset = 'train'):
        
        x = data.drop(columns = ['class'])
        y = data['class']
        plt.figure(figsize=(18, 16), dpi=300)
        sns.histplot(data=x) 
        plt.title("Histogram plot")
        plt.tight_layout()
        plt.xticks(rotation = 90)
        plt.yticks(rotation=0)
        plt.savefig(os.path.join(output_folder, f"{clf_name} {clf} hist plot {subset}.png")) 
        plt.close()


    """Empirical CDF plot"""
    def ecdf_plot(self, data, clf, clf_name, output_folder, subset = 'train'):
        plt.figure(figsize=(18, 16))
        x = data.drop(columns = ['class'])
        y = data['class']
        sns.ecdfplot(data=x)
        plt.title("ecdf plot")
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f'{clf_name} {clf} ecdf plot_{subset}.png'))
        plt.close()



