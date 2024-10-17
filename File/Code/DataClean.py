import pandas as pd
import numpy as np
from loguru import logger
import matplotlib.pyplot as plt
import json
import logging
from Data_Reading import DataLoader
from Data_Reading import loading_main
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore


class DataCleaner:

    def __init__(self, file_path):
        self.threshold = None
        if file_path is None:
            raise ValueError("Please provide a file path!!!!")
        else:
            file = pd.read_csv(file_path)
            self.data_frame = file
            self.cleaned_data_frame = None 

    """Handling Missing values."""
    def identify_missing_values(self):
        self.threshold = 100
        if self.data_frame.isnull().values.any():
            logger.info("Your data has some NaN values.")
            num_of_missing = self.data_frame.isnull().sum().sum()
            logger.info(f"The number of missing values: {num_of_missing}")
            num_of_missing_column = self.data_frame.isnull().sum()

            self.data_frame.dropna(axis=1, thresh=(len(self.data_frame) - self.threshold), inplace=True)             
            self.data_frame.dropna(axis=0, thresh=(len(self.data_frame.columns) - self.threshold), inplace=True)     
            if "Unnamed: 0" in self.data_frame.columns:
                self.data_frame.drop(columns=["Unnamed: 0", "Patient ID"], axis=1, inplace=True)

    """Identify and handle duplicate data by removing."""
    def identify_handle_duplicates(self):
        duplicate_count = self.data_frame.duplicated().sum()
        if duplicate_count > 0:
            self.data_frame.drop_duplicates(keep='first', inplace=True)
            logger.info(f"Removed {duplicate_count} duplicate values.")
        else:
            logger.info("No duplicate values found.")


    def remove_zero_columns_rows(self):
        """Removing columns with all values 0"""
        cols_to_remove = [col for col in self.data_frame.columns if self.data_frame[col].isin([0]).all()]
        self.data_frame.drop(columns=cols_to_remove, inplace=True)
        logger.info(f"Removed columns with all values 0: {cols_to_remove}")

        rows_to_remove = self.data_frame.index[self.data_frame.apply(lambda row: row.isin([0]).all(), axis=1)]
        self.data_frame.drop(index=rows_to_remove, inplace=True)
        logger.info(f"Removed rows with all values 0: {rows_to_remove.tolist()}")

    """Identify and Remove Outliers of data, based on chosen method (iqr or z-score)"""
    def identify_outliers(self, method='iqr', threshold=3):
        outliers = []
        numeric_columns = self.data_frame.select_dtypes(include=[np.number]).columns
        numeric_columns = [col for col in numeric_columns if col != 'class']
        
        for column in numeric_columns:
            if method == 'iqr':
                q1 = self.data_frame[column].quantile(0.25)
                q3 = self.data_frame[column].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                print(f'q1 is: {q1} and q3 is: {q3} and iqr is: {iqr}')                
                print(f'lower band is: {lower_bound} and upper band is: {upper_bound}') 
                outliers.extend(self.data_frame[(self.data_frame[column] < lower_bound) | (self.data_frame[column] > upper_bound)].index.tolist())

                outliers = list(set(outliers))
                with open(f'/File/Results/iqr_outliers.json', 'w') as file:
                    json.dump(outliers, file, indent=4)
                
                self.cleaned_data_frame = self.data_frame.drop(index=outliers)  ## NEW code
                return self.cleaned_data_frame          ### outliers deleted and self.cleaned_data_frame replaced

            
            elif method == "z-score":
                """Applies zscore on columns to get zscore"""
                z_scores = self.data_frame[numeric_columns].apply(zscore)

                """Identify outliers based on the Z-score threshold"""
                for column in numeric_columns:
                    column_outliers = self.data_frame[(np.abs(z_scores[column]) > threshold)].index.tolist()
                    outliers.extend(column_outliers)

                outliers = list(set(outliers))
                with open(f'/File/Results/z_score_outliers.json', 'w') as file:
                    json.dump(outliers, file, indent=4)
                    
                """Drop outliers from the data frame"""
                self.cleaned_data_frame = self.data_frame.drop(index=outliers)
                column_to_plot = self.cleaned_data_frame.iloc[:, 0:]
                return self.cleaned_data_frame  

    """A method to have minimum and maximum value of dataset"""
    def overall_data_range(self):
        numeric_columns = self.data_frame.select_dtypes(include=[np.number]).columns
        min_value = np.inf
        max_value = -np.inf

        for column in numeric_columns:
            column_min = self.data_frame[column].min()
            column_max = self.data_frame[column].max()

            if column_min < min_value:
                min_value = column_min
            if column_max > max_value:
                max_value = column_max
        return min_value, max_value

    """After cleaning data frame from different outliers, save in a path that is specified by user"""
    def get_cleaned_data(self, save_path):
        self.save_path = save_path
        self.cleaned_data_frame.to_csv(self.save_path, index=False)    
        logger.info("After cleaning, data saved in the path which you chose!!!")


def cleaner_main():
    logging.basicConfig(level=logging.INFO)
    file_path = f'/File/Results/first_checked.csv'
    cleaner = DataCleaner(file_path)
    cleaner.identify_missing_values()
    cleaner.identify_handle_duplicates()
    cleaner.remove_zero_columns_rows()
    cleaner.identify_outliers()
    overall_min, overall_max = cleaner.overall_data_range()
    print(f"Overall min value in your dataset is: {overall_min} and Overall max value is: {overall_max}")

    save_path = f'/File/Results/cleaned_data.csv'
    cleaner.get_cleaned_data(save_path)
    
