import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def load_and_filter_data(input_file):
    df = pd.read_csv(input_file, sep=';', encoding='utf-8', low_memory=False, index_col=0)
    missing_values = df.isna()
    valid_rows = np.logical_not(missing_values.any(axis=1))

    if not valid_rows.all():
        invalid_rows = np.where(np.logical_not(valid_rows))[0]
        for row in invalid_rows:
            print(f"Missing value in row {row} in {input_file}")
        df = df[valid_rows]
    return df

def save_valid_rows_to_combined_csv(input_files, combined_output_file):
    combined_data = pd.DataFrame()

    for input_file in input_files:
        print(input_file)
        df = load_and_filter_data(input_file)
        combined_data = pd.concat([combined_data, df])  # Concatenate selected rows
        print("Valid rows so far:", len(combined_data))

    combined_data.to_csv(combined_output_file, sep=';', encoding='utf-8')
    print(f"Valid rows with original header saved to {combined_output_file}")

def calculate_and_export_correlation(input_file, correlation_output_file):
    df = pd.read_csv(input_file, sep=';', encoding='utf-8', low_memory=False)
    selected_columns = df.filter(regex=r'^PUNT_', axis=1)

    correlation_pd = selected_columns.corr()
    print("Correlation using NumPy:")
    print(correlation_pd)

    correlation_np = np.corrcoef(selected_columns, rowvar=False)
    print("\nCorrelation using Pandas:")
    print(correlation_np)

    correlation_pd.to_csv(correlation_output_file, sep=';', encoding='utf-8')
    print(f"Correlation matrix saved to {correlation_output_file}")

def generate_heatmap(correlation_matrix, heatmap_image_file):
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.xticks(fontsize=6, rotation=45)
    plt.yticks(fontsize=6)
    plt.savefig(heatmap_image_file, dpi=300)
    plt.clf()
    print(f"Heatmap saved to {heatmap_image_file}")

def main():
    # Set the CSV folder path
    root = os.environ['root']
    csv_folder_path = os.path.join(root, "Csv")
    input_files = [os.path.join(csv_folder_path, file) for file in os.listdir(csv_folder_path) if file.endswith("_tr.CSV")]

    combined_output_file = os.path.join(csv_folder_path, "combined_output.csv")
    correlation_output_file = os.path.join(csv_folder_path, "combine_correlation.csv")
    heatmap_image_file = os.path.join(csv_folder_path, "combined_heatmap.png")

    if not os.path.exists(combined_output_file):
        save_valid_rows_to_combined_csv(input_files, combined_output_file)
    else:
        print(f"{combined_output_file} already exists.")
        combined_data = pd.read_csv(combined_output_file, sep=';', encoding='utf-8', index_col=0)

    if not os.path.exists(correlation_output_file):
        calculate_and_export_correlation(combined_output_file, correlation_output_file)
    else:
        print(f"{correlation_output_file} already exists.")
    
    correlation_matrix = pd.read_csv(correlation_output_file, sep=';', encoding='utf-8', index_col=0)

    if not os.path.exists(heatmap_image_file):
        generate_heatmap(correlation_matrix, heatmap_image_file)
    else:
        print(f"Heatmap {heatmap_image_file} already exists.")

if __name__ == "__main__":
    main()
