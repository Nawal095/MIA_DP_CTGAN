import pandas as pd
import numpy as np
import sys
import logging
from scipy.stats import entropy
from sklearn.metrics import jaccard_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def preprocess_data(file_path):
    logging.info(f'Preprocessing data from {file_path}')
    data = pd.read_csv(file_path)
    logging.info(f'Initial data shape: {data.shape}')
    label_encoders = {}
    for column in data.select_dtypes(include=['object']).columns:
        label_encoders[column] = LabelEncoder()
        data[column] = label_encoders[column].fit_transform(data[column])
        logging.info(f'Encoded column: {column}')
    logging.info(f'Preprocessed data shape: {data.shape}')
    return data

def calculate_kl_divergence(row1, row2):
    # Ensure the rows are valid probability distributions
    row1 = np.clip(row1, 1e-10, None)
    row2 = np.clip(row2, 1e-10, None)
    kl_div = entropy(row1, row2)
    logging.info(f'KL Divergence: {kl_div}')
    return kl_div

def calculate_jaccard_similarity(row1, row2):
    jaccard_sim = jaccard_score(row1, row2, average='macro')
    logging.info(f'Jaccard Similarity: {jaccard_sim}')
    return jaccard_sim

def plot_line_chart(kl_similarities, jaccard_similarities, method):
    plt.figure(figsize=(10, 6))
    if method in ['KL', 'Both']:
        plt.plot(kl_similarities, color='green', label='KL Divergence')
        plt.fill_between(range(len(kl_similarities)), kl_similarities, color='blue', alpha=0.1)
    if method in ['Jaccard', 'Both']:
        plt.plot(jaccard_similarities, color='orange', label='Jaccard Similarity')
        plt.fill_between(range(len(jaccard_similarities)), jaccard_similarities, color='yellow', alpha=0.25)
    plt.xlabel('Synthetic Rows')
    plt.ylabel('Score')
    if method == 'KL':
        plt.title(f'{method} Divergence')
    if method == 'Jaccard':
        plt.title(f'{method} Similarity')
    if method == 'Both':
        plt.title(f'{method} Jaccard Similarity and KL Divergence')
    plt.legend()
    plt.show()

def plot_radar_chart(kl_similarities, jaccard_similarities, method):
    categories = ['Row ' + str(i) for i in range(len(kl_similarities))]
    categories += categories[:1]  # Repeat the first category to close the circle
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    if method in ['KL', 'Both']:
        values = kl_similarities.tolist()
        values += values[:1]  # Repeat the first value to close the circle
        ax.plot(categories, values, color='green', linewidth=2, label='KL Divergence')
        ax.fill(categories, values, color='blue', alpha=0.1)
    if method in ['Jaccard', 'Both']:
        values = jaccard_similarities.tolist()
        values += values[:1]  # Repeat the first value to close the circle
        ax.plot(categories, values, color='orange', linewidth=2, label='Jaccard Similarity')
        ax.fill(categories, values, color='yellow', alpha=0.25)
    plt.title(f'{method} Radar Chart')
    plt.legend()
    plt.show()

def main(file1_path, file2_path, method, chart_type):
    logging.info(f'Starting main function with method: {method} and chart type: {chart_type}')
    
    # Preprocess the data
    file1 = preprocess_data(file1_path)
    file2 = preprocess_data(file2_path)

    # Drop the header row
    file1 = file1.iloc[1:]
    file2 = file2.iloc[1:]

    logging.info(f'File1 shape after dropping header: {file1.shape}')
    logging.info(f'File2 shape after dropping header: {file2.shape}')

    # Initialize the similarity/divergence vectors
    kl_similarities = []
    jaccard_similarities = []

    # Calculate similarity/divergence
    for i in range(len(file2)):
        row2 = file2.iloc[i].values
        kl_scores = []
        jaccard_scores = []
        for j in range(len(file1)):
            row1 = file1.iloc[j].values
            if method in ['KL', 'Both']:
                kl_score = calculate_kl_divergence(row2, row1)
                kl_scores.append(kl_score)
            if method in ['Jaccard', 'Both']:
                jaccard_score = calculate_jaccard_similarity(row2, row1)
                jaccard_scores.append(jaccard_score)
        if method in ['KL', 'Both']:
            kl_similarities.append(np.min(kl_scores))
            logging.info(f'Min KL Similarity for row {i}: {kl_similarities[-1]}')
        if method in ['Jaccard', 'Both']:
            jaccard_similarities.append(np.max(jaccard_scores))
            logging.info(f'Max Jaccard Similarity for row {i}: {jaccard_similarities[-1]}')

    # Convert to numpy arrays
    kl_similarities = np.array(kl_similarities)
    jaccard_similarities = np.array(jaccard_similarities)
    logging.info(f'KL Similarities array: {kl_similarities}')
    logging.info(f'Jaccard Similarities array: {jaccard_similarities}')

    # Plot the selected chart type
    if chart_type == 'line':
        plot_line_chart(kl_similarities, jaccard_similarities, method)
    elif chart_type == 'radar':
        plot_radar_chart(kl_similarities, jaccard_similarities, method)
    else:
        logging.error(f'Invalid chart type: {chart_type}')
        print("Invalid chart type. Please select either 'line' or 'radar'.")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python script.py <file1.csv> <file2.csv> <KL/Jaccard/Both> <line/radar>")
        sys.exit(1)

    file1_path = sys.argv[1]
    file2_path = sys.argv[2]
    method = sys.argv[3]
    chart_type = sys.argv[4]
    main(file1_path, file2_path, method, chart_type)
