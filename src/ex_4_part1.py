import numpy as np
import librosa
import os
from natsort import natsorted  # For natural sorting

def extract_mfcc(filepath):
    y, sr = librosa.load(filepath, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    return mfcc[:, :32]  # Ensure dimensions are (20, 32)

def dtw_distance(mfcc1, mfcc2):
    rows, cols = len(mfcc1[0]), len(mfcc2[0])
    dtw_matrix = np.full((rows + 1, cols + 1), np.inf)
    dtw_matrix[0, 0] = 0

    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            cost = np.linalg.norm(mfcc1[:, i-1] - mfcc2[:, j-1])  # Euclidean distance
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j],      # Insertion
                dtw_matrix[i, j-1],      # Deletion
                dtw_matrix[i-1, j-1]     # Match
            )
    return dtw_matrix[rows, cols]

def classify(test_mfcc, train_data, method='dtw'):
    min_distance = float('inf')
    predicted_label = None

    for label, mfcc_list in train_data.items():
        for train_mfcc in mfcc_list:
            if method == 'dtw':
                distance = dtw_distance(test_mfcc, train_mfcc)
            elif method == 'euclidean':
                distance = np.linalg.norm(test_mfcc - train_mfcc)
            if distance < min_distance:
                min_distance = distance
                predicted_label = label

    return predicted_label

def generate_output(test_files, train_data, output_file):
    # Sort test files in natural order (e.g., "sample1.wav", "sample2.wav", ...)
    test_files = natsorted(test_files)

    # Map from string labels to numeric labels
    label_mapping = {'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5'}

    with open(output_file, 'w') as f:
        for test_file in test_files:
            test_mfcc = extract_mfcc(test_file)
            pred_euclidean = classify(test_mfcc, train_data, method='euclidean')
            pred_dtw = classify(test_mfcc, train_data, method='dtw')

            # Map labels to numeric format before writing to output
            pred_euclidean = label_mapping[pred_euclidean]
            pred_dtw = label_mapping[pred_dtw]

            # Write to output file
            f.write(f"{os.path.basename(test_file)} - {pred_euclidean} - {pred_dtw}\n")

def load_train_data(train_dir):
    train_data = {}
    for label in os.listdir(train_dir):
        label_dir = os.path.join(train_dir, label)
        if os.path.isdir(label_dir):  # Ensure it's a directory
            train_data[label] = []
            for file in os.listdir(label_dir):
                if file.endswith('.wav'):
                    filepath = os.path.join(label_dir, file)
                    train_data[label].append(extract_mfcc(filepath))
    return train_data

# Get list of test files
test_files = [os.path.join('test_files', file) for file in os.listdir('test_files') if file.endswith('.wav')]

# Load training data
train_data = load_train_data('train_data')

# Generate output
generate_output(test_files, train_data, 'output.txt')
