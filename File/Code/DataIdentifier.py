import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import logging
import matplotlib.pyplot as plt

""" To identify different types of data, like numerical and categorical."""
class DataIdentifier:

    def __init__(self, file_path):
        if file_path is None:
            raise ValueError("Please provide a file path!!!!")
        else:
            file_path = pd.read_csv(file_path)
            self.data_frame = file_path

    def detect_numeric_columns(self):
        """Detect numeric columns in the DataFrame."""
        return self.data_frame.select_dtypes(include=["number"]).columns.tolist()

    def detect_categorical_columns(self):
        """Detect categorical columns in the DataFrame."""
        return self.data_frame.select_dtypes(include=["object"]).columns.tolist()

    def transform_categorical_columns(self):
        """Transform categorical columns using OneHotEncoder."""
        categorical_cols = self.detect_categorical_columns()
        encoder = OneHotEncoder(sparse=False)
        for col in categorical_cols:
            transformed_data = encoder.fit_transform(self.data_frame[[col]])

            """Create column names for each category"""
            col_names = [f"{col}_{category}" for category in encoder.categories_[0]]

            """Replace the original column with new columns in the DataFrame"""
            self.data_frame = self.data_frame.drop(columns=[col])
            self.data_frame = pd.concat([self.data_frame, pd.DataFrame(transformed_data, columns=col_names)], axis=1)


    """For ordinal method, I should know what ranges I have. this method is just as an exam until I have the range"""
    def detect_ordinal_columns(self):
        """Detecting ordinal columns in DataFrame"""
        ordinal_columns = []
        for col in self.data_frame.columns:
            if "we can give here the pattern of ordinal data" in col:
                ordinal_columns.append[col]
        return ordinal_columns

    def transform_ordinal_columns(self, ordinal_mapping):
        """Transforming ordinal columns based on a map which we define """
        for col, mapping in ordinal_mapping.items():
            if col in self.data_frame.columns:
                self.data_frame[col] = self.data_frame[col].map(mapping)


    def converted_data(self, save_path):
        self.save_path = save_path
        self.data_frame.to_csv(self.save_path, index=False)
        print("The data is checked for Categorical and Other objective data type!!!!")


def Identifier_main():
    logging.basicConfig(level = logging.INFO)
    file_path = f'/File/Results/cleaned_data.csv'
    identifier = DataIdentifier(file_path)
    identifier.detect_numeric_columns()
    identifier.detect_categorical_columns()
    identifier.detect_ordinal_columns()
    save_path = f'/File/Results/final_file.csv'
    identifier.converted_data(save_path)
    # As an example of how to call ordinal method
    # ordinal_mappings = {
    #     "ordinal_column1": {"Poor": 1, "Fair": 2, "Good": 3, "Very Good": 4, "Excellent": 5},
    # } 
    # cleaner.transform_ordinal_columns()
