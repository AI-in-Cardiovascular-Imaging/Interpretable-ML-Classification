import import_ipynb
import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier
from lime import lime_tabular
from interpret import show
from interpret.blackbox import PartialDependence, LimeTabular, ShapKernel, MorrisSensitivity
from interpret.data import ClassHistogram
from interpret.perf import ROC
from sklearn.linear_model import LogisticRegression
from pdpbox import pdp, info_plots
import matplotlib.pyplot as plt
import shap
import eli5
from eli5.sklearn import PermutationImportance
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from IPython.display import display
import interpret
from interpret import show
import joblib
shap.initjs()
shap.plots.initjs()
from alibi.explainers import PartialDependenceVariance, plot_pd_variance


class Interpret:
    def __init__(self, method):
        self.shap_values = None
        self.method = method
    
    def load(self):
        """pkl load here"""
        # train = pd.read_csv("F:\\Final Results\\model output\\RandomSearch_Combined_Data_Z_score\\LogisticRegression(max_iter=10000, tol=0.1)\\Lr Select from Model selected features train.csv")     # Combined_zscore_randomsearch
        # train = pd.read_csv("F:\\Final Results\\model output\\Random Search_Zscore_Stress Data\\KNeighborsClassifier()\\Knn Select from Model selected features train.csv")         # Stress_zscor_Randomsearch
        train = pd.read_csv("F:\\Final Results\\model output\\Random Search_IQR_Rest Data 2\\LogisticRegression(max_iter=10000, tol=0.1)\\Lr Pca selected features train.csv")          # Rest_iqr_RandomSearch
        X_train = train.drop(columns=['class'])
        y_train = train['class']
        self.X_train = X_train
        self.y_train = y_train

        # test = pd.read_csv("F:\\Final Results\\model output\\RandomSearch_Combined_Data_Z_score\\LogisticRegression(max_iter=10000, tol=0.1)\\Lr Select from Model selected features test.csv")       # Combined
        # test = pd.read_csv("F:\\Final Results\\model output\\Random Search_Zscore_Stress Data\\KNeighborsClassifier()\\Knn Select from Model selected features test.csv")           # Stress_zscore_Randomsearch
        test = pd.read_csv("F:\\Final Results\\model output\\Random Search_IQR_Rest Data 2\\LogisticRegression(max_iter=10000, tol=0.1)\\Lr Pca selected features test.csv")           # Rest_iqr_RandomSearch
        X_test = test.drop(columns=['class'])
        y_test = test['class']
        self.X_test = X_test
        self.y_test = y_test

    def model_read(self, model):
        self.load()
        self.model = model
        self.model.fit(self.X_train, self.y_train)
        return self.model

    """self.explainer = shap.TreeExplainer(self.model, shap.sample(self.X_train, 100))      # Tree-based algorithms"""    
    def Explainer(self):

        methods = ['shap', 'Kernel_shap', 'Partition_shap', 'Permutation_shap']
        if self.method == 'shap':
            self.explainer = shap.Explainer(self.model.predict, shap.sample(self.X_train, 35)) 
            
            shap_values = self.explainer(self.X_test)   # Calculating Shap_values
            self.shap_values = shap_values
            print("The shape of Shap values : ", shap_values.shape)

            save_shap_values = np.array(shap_values)
            np.save("D:\\Thesis\\SHAP_VALUES.npy", shap_values.values)   # Save SHAP values to numpy array

            """Calculating Expected Value"""
            self.expected_value = self.y_train.mean()                   
            print("The Expected Value is: ", self.expected_value)

            """Find the feature with highest shap value"""
            max_shap_feature_index = np.argmax(np.abs(self.shap_values.values).mean(0))
            self.max_shap_feature = self.X_test.columns[max_shap_feature_index]
            print("Feature with Maximum Shap Value is: ", self.max_shap_feature)
            
            
        elif self.method == 'shap_kernel':
            self.explainer = shap.KernelExplainer(self.model.predict, shap.sample(self.X_train, 35))
            shap_values = self.explainer(self.X_test)   # Calculating Shap_values
            self.shap_values = shap_values
            print("The shape of Shap values : ", shap_values.shape)

            save_shap_values = np.array(shap_values)
            np.save("D:\\Thesis\\SHAP_VALUES.npy", shap_values.values)   # Save SHAP values to numpy array

            """Calculating Expected Value"""
            self.expected_value = self.y_train.mean()                   
            print("The Expected Value is: ", self.expected_value)

            """Find the feature with highest shap value"""
            max_shap_feature_index = np.argmax(np.abs(self.shap_values.values).mean(0))
            self.max_shap_feature = self.X_test.columns[max_shap_feature_index]
            print("Feature with Maximum Shap Value is: ", self.max_shap_feature)
            
        
        elif self.method == 'lime':
            lime_explainer = lime_tabular.LimeTabularExplainer(self.X_train.values,
                                                   feature_names=self.X_train.columns, feature_selection='auto',
                                                   mode='classification')
            instance_index = 1  
            explanation = lime_explainer.explain_instance(self.X_test.iloc[instance_index], 
                                                           self.model.predict_proba,
                                                           num_features=len(self.X_test.columns))
            
    
    """Different plots for shap values"""
    def plot_part(self):

        shap.violin_plot(self.shap_values, features=self.X_test, feature_names=self.X_test.columns)             # Violin plot for shap values
#         shap.violin_plot(self.shap_values)     # Original code
        
        shap.plots.bar(self.shap_values, max_display=20, show_data='auto', show=True)   # Bar plot for shap values
    
        """Waterfall plot, visualize the first prediction's explanation"""
        shap.waterfall_plot(self.shap_values[0], max_display=20)

        """Summary plot and Beeswarm plot which officially are same!!!!"""
        shap.summary_plot(self.shap_values, features=self.X_test, feature_names=self.X_test.columns) # feature_names=self.X_test.columns added

        shap.plots.beeswarm(self.shap_values, max_display=20, color=plt.get_cmap("cool"))

        """Basic PDP plot of the feature with Max shap value"""
        shap.partial_dependence_plot(self.max_shap_feature, self.model.predict, self.X_test,
                                    model_expected_value=True, feature_expected_value=True)                 # 'Grey level non-uniformity.12'
        
        """Create a heatmap plot of a set of SHAP values."""
        shap.plots.heatmap(self.shap_values, max_display=20)      
        
        # Create a SHAP dependence scatter plot, colored by an interaction feature.
        shap.plots.scatter(self.shap_values[:, self.max_shap_feature], color = self.shap_values, alpha=0.6)     # scatter plot for highest value of Shap value
        
        """a scatter plot that we don't know the name of feature, and we just want to plot the important feature plot"""
        # shap.plots.scatter(self.shap_values[:, self.shap_values.abs.mean(0).argsort[-1]])
        
        """Scatter plot of all featues by shap_values"""
        shap.plots.scatter(self.shap_values, ylabel="SHAP value\n(higher means more likely)")  
        plt.xticks(rotation=90)
        plt.show()


        """Decision Plot of Shap_Values"""
        range1= range(len(self.X_test))
        shap_values_array = self.shap_values.values if isinstance(self.shap_values, shap.Explanation) else self.shap_values
        shap.decision_plot(self.expected_value, shap_values_array[range1], features=self.X_test, feature_names=self.X_test.columns.tolist()) # , link="logit"

        
        """Force Plot. Visualize the given SHAP values with an additive force layout."""
        base_value = self.shap_values.base_values
        if self.shap_values is not None:
            shap.force_plot(base_value, self.shap_values.values[:1, :], self.X_test.iloc[:1, :], matplotlib=True)
            plt.xticks(rotation=90, fontsize = 8)
        else:
            print("SHAP values are not available. Please run the Explainer method first.")

            
    """pdp plot for one specific feature(max shap value)"""
    def plot_pdp(self):
        pdp_goal = pdp.pdp_isolate(model=self.model, dataset=self.X_test, model_features=self.X_test.columns.tolist() ,feature=self.max_shap_feature)                                                    # instead of 'Grey level non-uniformity.12', we put self.max_shap_feature
        pdp.pdp_plot(pdp_goal, self.max_shap_feature)       # we put self.max_shap_feature
        plt.show()
        
    """Provides histogram visualizations for classification problems."""
    def histogram_plot(self):
        train_histogram = ClassHistogram().explain_data(self.X_train, self.y_train, name='Train Data Explaining')
        show(train_histogram)

        test_histogram = ClassHistogram().explain_data(self.X_test, self.y_test, name='Test Data Explaining')
        show(test_histogram)
        
    def permutation_importance(self):
        perm = PermutationImportance(self.model, random_state=1).fit(self.X_train, self.y_train)
        return eli5.show_weights(perm, feature_names = self.X_test.columns.tolist())
    
    """Partial dependence plots visualize the dependence between the response and a set of target features"""
    def pdp_of_features(self):
        pdp = PartialDependence(self.model, self.X_train)
        show(pdp.explain_global(), 0)
        plt.show()
        
    def lime_explain(self):
        self.X_train=self.X_train.astype(float)
        self.X_test = self.X_test.astype(float)
        self.y_test = self.y_test.astype(int)
        self.y_train = self.y_train.astype(int)
        lime = LimeTabular(self.model, self.X_train, random_state=4)
        show(lime.explain_local(self.X_test[:15], self.y_test[:15]), 0)
    
    """Exposes SHAP kernel explainer from shap package, in interpret API form."""
    def shap_kernel(self):
        shap = ShapKernel(self.model, self.X_train)
        shap_local = shap.explain_local(self.X_test[:10], self.y_test[:10])
        show(shap_local, 0)
        
    def Moris_Sensitivity(self):
        msa = MorrisSensitivity(self.model, self.X_train)
        show(msa.explain_global())

    def Moris_Sensitivity_global(self):
        sensitivity = MorrisSensitivity(self.model, self.X_test)
        sensitivity_global = sensitivity.explain_global(name="Global Sensitivity")
        show(sensitivity_global)
    
    def ROC_plot(self):
        blackbox_perf = ROC(self.model).explain_perf(self.X_train, self.y_train, name='ROC_plot_Train')
        show(blackbox_perf)
        
        blackbox_perf = ROC(self.model).explain_perf(self.X_test, self.y_test, name='ROC_plot_Test')
        show(blackbox_perf)

        

    
def interpretability_main():
    
    # file_path = "F:\\Final Results\\pkl\\RandomSearch_Combined_data_Z_score pkl\\Lr Select from Model RobustScaler best_model.pkl"      # Combined
    # file_path = "F:\\Final Results\\pkl\\RandomSearch_Zscore_Stress pkl\\Knn Select from Model MinMax best_model.pkl"                      # Stress
    file_path = "F:\\Final Results\\pkl\\Random Search_IQR_Rest pkl2\\Lr Pca Standard best_model.pkl"                               # Rest_iqr_randomsearch

    with open(file_path, 'rb') as file:
        model = joblib.load(file)

    interpretability = Interpret(method='shap')
    interpretability.model_read(model=model)
    interpretability.Explainer()
    interpretability.plot_part()
    interpretability.plot_pdp()
    interpretability.pdp_of_features()
    permutation_importance_result = interpretability.permutation_importance()
    display(permutation_importance_result)
    interpretability.histogram_plot()
    interpretability.lime_explain()
    interpretability.shap_kernel()
    interpretability.Moris_Sensitivity()
    interpretability.Moris_Sensitivity_global()
    interpretability.ROC_plot()


if __name__=="__main__":
    interpretability_main()