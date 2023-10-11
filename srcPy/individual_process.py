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
    selected_columns = df.filter(regex=r'^(P|p)(U|u)(N|n)(T_|t_)', axis=1)
    
    missing_values = selected_columns.isna()
    valid_rows = np.logical_not(missing_values.any(axis=1))  # Find rows without missing values

    if not valid_rows.all():  # Check if there are invalid rows
        invalid_rows = np.where(np.logical_not(valid_rows))[0]
        for row in invalid_rows:
            print(f"Missing value in row {row} in {input_file}")
        selected_columns = selected_columns[valid_rows]  # Filter out rows with missing values
    else:
        print("There is no missing data in punt_")

    correlation_pd = selected_columns.corr()
    print("Correlation using NumPy:")
    print(correlation_pd)

    correlation_np = np.corrcoef(selected_columns, rowvar=False)
    print("\nCorrelation using Pandas:")
    print(correlation_np)

    correlation_pd.to_csv(output_file, sep=';', encoding='utf-8')


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
        else:
            print(filename + " correlation file exists.")
            
        # Define the filename for the heatmap image
        heatmap_image_file = os.path.join(csv_folder_path, filename.replace(".TXT", "_heatmap.png").replace(".txt", "_heatmap.png"))
        
        # Check if the heatmap image file exists
        if not os.path.exists(heatmap_image_file):
            # Load the correlation matrix as a DataFrame
            correlation_matrix = pd.read_csv(correlation_output_file, sep=';', encoding='utf-8', index_col=0)
            
            # Create a heatmap
            plt.figure(figsize=(12, 10))
            sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
            plt.title("Correlation Matrix")
            plt.xticks(fontsize=6, rotation=45)
            plt.yticks(fontsize=6)
            
            # Save the heatmap as an image
            plt.savefig(heatmap_image_file, dpi=300)
            
            # Clear the current plot
            plt.clf()
            
            print("Heatmap for", filename, "has been created and saved.")
        else:
            print("Heatmap for", filename, "already exists.")

print("Correlation calculation and heatmap generation completed.")
