import argparse
import subprocess
import sys
from functions import (
    lp_distance, cosine_distance, maha2_distance,
    maha_distance, distancemat, bygroup, kn,
    final_groups
)
import os
import numpy as np
import functools
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder

def main(file_path):
    # Get the file extension to determine the file type
    rest, file_extension = os.path.splitext(file_path)

    # Load the data based on file extension
    if file_extension == '.xlsx':
        data = pd.read_excel(file_path)
    elif file_extension == '.csv':
        data = pd.read_csv(file_path)
    elif file_extension == '.txt':
        data = pd.read_table(file_path)  # Assumes tab-delimited, adjust as necessary
    elif file_extension == '.json':
        data = pd.read_json(file_path)
    else:
        data = pd.DataFrame(load_iris().data)
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns

    # Apply One-Hot Encoding to categorical columns
    if not categorical_cols.empty:
        encoder = OneHotEncoder(sparse_output=False, drop = "first", handle_unknown='ignore')
        encoded_data = encoder.fit_transform(data[categorical_cols])
        encoded_data = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_cols))

        # Drop original categorical columns and concatenate encoded data
        data = data.drop(columns=categorical_cols)
        data = pd.concat([data, encoded_data], axis=1)
    data.fillna(data.mean(), inplace=True)
    # scatter_matrix(data, alpha=0.2, figsize=(100, 100), diagonal='kde')
    # plt.show()
    data = data.values
    min_value = np.min(data, axis=0)
    max_value = np.max(data, axis=0)
    data = (data - min_value) / (max_value - min_value)
    # Define the partial function for lp_distance with p=4
    lp_distance_pinf = functools.partial(lp_distance, p=np.inf)

    # Mahalanobis distance requires a different function call, so we'll handle it separately
    maha_distance_matrix = distancemat(data, distance=maha_distance)

    # Now call distancemat with your data and the partial function for lp distance with p=np.inf
    lp_distance_matrix_pinf = distancemat(data, distance=lp_distance_pinf)

    # Set the parameters for final classification
    k = 50
    g = 5
    n_iterations = 1

    # Perform final classification for both Mahalanobis and LP norm with p=np.inf
    final_labels_maha = final_groups(maha_distance_matrix, kn, k, n_iterations)
    final_labels_lp_pinf = final_groups(lp_distance_matrix_pinf, bygroup, g, n_iterations)
    return final_labels_maha, final_labels_lp_pinf

if __name__ == '__main__':
    # Create a virtual environment
    subprocess.run([sys.executable, "-m", "venv", "env"])

    # Activate the virtual environment
    # On Windows:
    activate_script = "env\\Scripts\\activate.bat"
    # On Unix or MacOS:
    # activate_script = "source env/bin/activate"
    subprocess.run([activate_script], shell=True)

    # Install dependencies from requirements.txt
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

    parser = argparse.ArgumentParser(description="Process a data file.")
    parser.add_argument('--file_path', type=str, default='', help='The path to the data file. Defaults to the Iris dataset if not provided.')
    args = parser.parse_args()

    labels = main(args.file_path)
    # Print or use the labels as needed
    print(labels)