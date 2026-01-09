import pickle
import pandas as pd
import matplotlib.pyplot as plt
import os
#from src.classifyre import classify_with_classyfire
#from src.big_file import big_file_classifier
from src.evaluation import save_results_histogram
from src.histogram_and_classes import get_class_dict_and_histogram
from src.training_functions import training

if __name__ == "__main__":
    # Parameters:
    train_using_classes = False

    # classify_with_classyfire()
    # big_file_classifier()

    if train_using_classes:
        class_dict = get_class_dict_and_histogram(1000) # devuelve compuestos por clase y nombre
    else:
        # Train with all smrt:
        df = pd.read_csv('resources/smrt_fingerprints.csv')
        class_dict = {'all': (len(df['inchi']), df['inchi'].tolist())}

    results_df = training(class_dict)
    print(results_df)
    save_results_histogram(results_df)

    # classification_elena()
    # training_elena()
