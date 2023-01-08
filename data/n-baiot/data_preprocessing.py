import os
import sys

import pandas as pd

if __name__ == '__main__':
    file_dir = "raw"
    assert os.path.isdir(file_dir), "There is no datasets"
    if os.path.exists(os.path.join(file_dir, 'data_set')):
        print("The data_set folder already exists")
        sys.exit(0)
    all_data = None
    for directory in os.listdir(file_dir):
        if directory != "data_set" and os.path.isdir(os.path.join(file_dir, directory)):
            print("Labeling and integrating "+directory)
            sub_directories = os.listdir(os.path.join(file_dir, directory))
            for sub_directory in sub_directories:
                if os.path.isdir(os.path.join(file_dir, directory, sub_directory)):
                    for csv_file in os.listdir(os.path.join(file_dir, directory, sub_directory)):
                        data = pd.read_csv(os.path.join(file_dir, directory, sub_directory, csv_file))
                        data.insert(data.shape[1], 'label', 1)
                        all_data = pd.concat([all_data, data], ignore_index=True)
                elif sub_directory == "benign_traffic.csv":
                    data = pd.read_csv(os.path.join(file_dir, directory, sub_directory))
                    data.insert(data.shape[1], 'label', 0)
                    all_data = pd.concat([all_data, data], ignore_index=True)
            print("saving " + directory)
            if not os.path.exists(file_dir + '/' + "data_set/"):
                os.makedirs(file_dir + '/' + "data_set/")
            loc = file_dir + '/' + "data_set/" + directory + '.csv'
            all_data.to_csv(loc, header=False)
            all_data = None

    print("Complete the classification and save")


