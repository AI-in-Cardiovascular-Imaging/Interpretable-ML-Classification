import import_ipynb
from Search import main_search
from Data_Reading import loading_main
from DataClean import cleaner_main
from DataIdentifier import Identifier_main
from Interpretability import interpretability_main
import pandas as pd

if __name__=="__main__":

    """Insert the file path"""
    file_path = pd.read_csv('/File/Files/df_rest.csv')                           
    # file_path = pd.read_csv('/File/Files//df_stress.csv')
    # file_path = pd.read_csv('/File/Files/merge_of_all_data.csv')
    loading_main(file_path)
    cleaner_main()
    Identifier_main()
    main_search()
    # interpretability_main()       # To have different SHAP plots

