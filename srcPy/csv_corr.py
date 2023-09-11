import os
import subprocess
import zipfile
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

root = os.environ['root']
zip_folder_path = root + "OneDrive_1_9-9-2023"
txt_folder_path = root + "Txt"
zip_files = [f for f in os.listdir(zip_folder_path) if f.endswith(".zip")]

csv_folder_path = root + "Csv"
os.makedirs(csv_folder_path, exist_ok=True)

# Function to calculate and export correlation matrix
def calculate_and_export_correlation(input_file, output_file):
    df = pd.read_csv(input_file, sep=';', encoding='utf-8', low_memory=False)
    selected_columns = df.filter(regex=r'^PUNT_', axis=1).columns
    selected_data = df[selected_columns].select_dtypes(include=['number'])
    correlation_matrix = selected_data.corr()
    correlation_matrix.to_csv(output_file, sep=';', encoding='utf-8')

# Function to normalize correlation matrix
def normalize_correlation_matrix(correlation_matrix):
    np.fill_diagonal(correlation_matrix.values, 1)
    max_val = np.nanmax(correlation_matrix.values)
    min_val = np.nanmin(correlation_matrix.values)
    
    # Check if max and min values are not NaN
    if not np.isnan(max_val) and not np.isnan(min_val):
        correlation_matrix = (correlation_matrix - min_val) / (max_val - min_val)
    else:
        # If max and min values are NaN, set all values to 0 (or any desired default value)
        correlation_matrix = correlation_matrix.fillna(0)
    
    return correlation_matrix

# Create a list to store the paths of all processed CSV files
processed_csv_files = []

for zip_file in zip_files:
    zip_file_path = os.path.join(zip_folder_path, zip_file)
    
    # Check if the TXT file has already been extracted
    txt_file_name = zip_file.replace(".zip", ".TXT")
    txt_file_name_txt = zip_file.replace(".zip", ".txt")
    if txt_file_name not in os.listdir(txt_folder_path) and txt_file_name_txt not in os.listdir(txt_folder_path):
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(txt_folder_path)

for filename in os.listdir(txt_folder_path):
    if filename.endswith(".TXT") or filename.endswith(".txt"):
        input_file = os.path.join(txt_folder_path, filename)
        output_file = os.path.join(csv_folder_path, filename.replace(".TXT", "_tr.CSV").replace(".txt", "_tr.CSV"))

        # Check if the corresponding CSV file already exists
        if not os.path.exists(output_file):
            subprocess.run(["tr", "¬¨", ";"], input=open(input_file, "rb").read(), stdout=open(output_file, "wb"))
        else:
            print(filename + " file exists.")

        # Define the correlation output file name
        correlation_output_file = os.path.join(csv_folder_path, filename.replace(".TXT", "_correlation.CSV").replace(".txt", "_correlation.CSV"))

        # Check if the correlation output file already exists
        if not os.path.exists(correlation_output_file):
            # Calculate and export correlation matrix
            calculate_and_export_correlation(output_file, correlation_output_file)
            
            # normalization
            correlation_matrix = pd.read_csv(correlation_output_file, sep=';', encoding='utf-8', index_col=0)
            normalized_correlation_matrix = normalize_correlation_matrix(correlation_matrix)
            normalized_correlation_matrix.to_csv(correlation_output_file, sep=';', encoding='utf-8')
        else:
            print(filename + " correlation file exists.")

        # Append the processed CSV file path to the list
        processed_csv_files.append(correlation_output_file)
        print("***", processed_csv_files)

# Combine the first four normalized correlation matrices
combined_correlation_matrix = None
for correlation_output_file in processed_csv_files:
    correlation_matrix = pd.read_csv(correlation_output_file, sep=';', encoding='utf-8', index_col=0)
    correlation_matrix.head()
    if combined_correlation_matrix is None:
        combined_correlation_matrix = correlation_matrix
    else:
        combined_correlation_matrix += correlation_matrix

# Calculate the combined correlation matrix for the first four datasets
combined_correlation_matrix /= len(processed_csv_files)

# 歸一化整體相關性矩陣
combined_correlation_matrix = normalize_correlation_matrix(combined_correlation_matrix)

# Define the combined correlation output file
combined_correlation_output_file = os.path.join(csv_folder_path, "combined_correlation_normalized.CSV")

# Export the combined correlation matrix to a CSV file
combined_correlation_matrix.to_csv(combined_correlation_output_file, sep=';', encoding='utf-8')

plt.figure(figsize=(12, 10))
sns.heatmap(combined_correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix")
plt.xticks(fontsize=6, rotation=45)
plt.yticks(fontsize=6)
plt.savefig(os.path.join(csv_folder_path, "correlation_heatmap.png"), dpi=300)  # Save as PNG image

print("Character replacement and normalized correlation calculation completed.")
