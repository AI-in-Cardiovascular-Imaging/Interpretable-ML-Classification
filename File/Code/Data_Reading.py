import os
import pandas as pd
import numpy as np
from loguru import logger
import logging
import seaborn as sns 
import matplotlib.pyplot as plt


class DataLoader:
    """Reads Excel, CSV, or DataFrame and returns a DataFrame."""
    def __init__(self, file_path):
        self.file = file_path
        self._frame = None
        self.saved_path = None
    
    def __call__(self):
        return self.read_file()

    def _set_frame(self, frame: pd.DataFrame):
        
        if not isinstance(frame, pd.DataFrame):
            logger.error(f'Input is not a DataFrame: {type(frame)}')
            return
        self._frame = frame
        logger.trace(f'Frame set -> {type(frame)}')


    """New def to read different file types."""
    def read_file(self):
        """Read file with different formats and returns a DataFrame."""
        if self._frame is not None:
            return self._frame

        try:
            if isinstance(self.file, pd.DataFrame):
                self._set_frame(self.file)
                logger.info('DataFrame provided directly')
                return self._frame

            if isinstance(self.file, str):
                if self.file.endswith(".csv"):
                    frame = pd.read_csv(self.file)
                    self._set_frame(frame)

                elif self.file.endswith(".xlsx"):
                    frame = pd.read_excel(self.file)
                    self._set_frame(frame)

                else:
                    logger.error(f'Invalid file type. Allowed types are .csv and .xlsx. Check -> {self.file}')
                    return None

                logger.info(f'Loaded file: {self.file}')
                return self._frame

            logger.error(f'Invalid input type. Expected a file path or DataFrame, got {type(self.file)}.')
            return None

        except FileNotFoundError:
            logger.error(f'File not found: {self.file}')
            return None
        except pd.errors.EmptyDataError:
            logger.error(f'No data: {self.file} is empty')
            return None
        except pd.errors.ParserError:
            logger.error(f'Parsing error: Could not parse {self.file}')
            return None
        except Exception as e:
            logger.error(f'Error reading file: {e}')
            return None


    def save_to_path(self, save_path=None):
        if self._frame is None:
            logging.error('No DataFrame to save.')
            return None
        
        if save_path is None:
            save_path = f'/File/Results/first_checked.csv'
            
        self._frame.to_csv(save_path, index=False)
        logging.info(f'Saved DataFrame to {save_path}')
        self.saved_path = save_path
        return save_path


def loading_main(file_path):
    logging.basicConfig(level = logging.INFO)    

    # file_path = pd.read_csv('/File/Files/df_rest.csv')                           
    # file_path = pd.read_csv('/File/Files//df_stress.csv')
    # file_path = pd.read_csv('/File/Files/merge_of_all_data.csv')
    file_path = file_path

    loader = DataLoader(file_path)
    df = loader()
    if df is not None:
        print("The size of your dataframe is:\n", df.shape)
        print("Checked file is stored in: ", loader.saved_path)
        print("The head of your data is:\n", df.head())
        print("DataFrame Describe is: \n", df.describe())         
    loader.save_to_path()

