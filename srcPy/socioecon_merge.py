import os
import glob
import pandas as pd

root = os.environ['ROOT']
folder_path = root + 'OneDrive_1_11-26-2023'

def delete_files_with_part2(directory):
    for root, dirs, files in os.walk(directory):
        print('root: ', root)
        for file in files:
            print('name: ', file)
            if file.lower().endswith('.csv') and 'part2' in file:
                file_path = os.path.join(root, file)
                print(f"Deleting file: {file_path}")
                os.remove(file_path)

# delete_files_with_part2(folder_path)

basic_path = root + '/Csv_11-2-2023/basic_file_output.csv'
merged_basic_df = pd.read_csv(basic_path, sep=',', encoding='utf-8', low_memory=False)
alarm = len(merged_basic_df['cod_departamento'])
# print("Just Check:\n", merged_basic_df.columns)
for root, dirs, files in os.walk(folder_path):
    for file in files:
        # if file == 'Unmet Basic Needs (NBI) per category %.csv':
        #     print('This is ', file, ' I\'m going to skip')
        #     continue
        # print('LOOK AT ME: ', file)
        if file.lower().endswith('.csv'):
            merge_file_path = os.path.join(root, file)
            open_merge_df= pd.read_csv(merge_file_path, sep=',', encoding='utf-8', low_memory=False, index_col=False)
            open_merge_df.columns = open_merge_df.columns.str.lower()
            print('Before everything:\n', open_merge_df.head(n=5))
            if 'clase' in open_merge_df.columns:
                open_merge_df['clase'] = open_merge_df['clase'].astype(str)
                clase_filter = open_merge_df['clase'].str.lower() == 'total'
                open_merge_df = open_merge_df[clase_filter]
                print('After filter:\n', open_merge_df.head(n=5))
            if 'anio' in open_merge_df.columns:
                numeric_columns = open_merge_df.select_dtypes(include='number').columns.drop(['cod_departamento','anio'])
                non_numeric_columns = open_merge_df.select_dtypes(exclude='number').columns.drop(['departamento'])
            else:
                numeric_columns = open_merge_df.select_dtypes(include='number').columns.drop(['cod_departamento'])
                non_numeric_columns = open_merge_df.select_dtypes(exclude='number').columns.drop(['departamento'])
        # Group by 'cod_departamento' and calculate the mean for each numeric column
            print('numeric_columns\n', numeric_columns)
            print('non_numeric_columns\n', non_numeric_columns)

            prefix = file.replace('.csv', '').replace(' ', '_').lower() + '_'
            if not numeric_columns.empty:
                open_merge_df[numeric_columns] = open_merge_df[numeric_columns].astype(float)
                mean_by_cod_departamento_df = open_merge_df.groupby('cod_departamento')[numeric_columns].mean().reset_index()
                print("Final:\n", mean_by_cod_departamento_df, "\n")
                mean_by_cod_departamento_df.rename(columns=lambda col: f"{prefix}{col}" if col in numeric_columns else col, inplace=True)
                # Set the display format for float values
                # pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))
                print("Mean by cod_departamento (formatted):\n", mean_by_cod_departamento_df)
                merged_basic_df = pd.merge(merged_basic_df, mean_by_cod_departamento_df, on='cod_departamento', how='left')

            if not non_numeric_columns.empty:
                non_numeric_temp = open_merge_df.select_dtypes(exclude='number').columns.drop(['departamento']).union(['cod_departamento'])
                non_numeric_df = open_merge_df[non_numeric_temp].drop_duplicates()
                print('non_numeric_df\n', non_numeric_df)
                non_numeric_df.rename(columns=lambda col: f"{prefix}{col}" if col in non_numeric_columns else col, inplace=True)
                print("Non_numeric:\n", non_numeric_df, "\n")
                merged_basic_df = pd.merge(merged_basic_df, non_numeric_df, on='cod_departamento', how='left')
            
            print("Ultimate:\n",merged_basic_df.head(n=5))
            print("Ultimate:\n",merged_basic_df.columns)


merged_basic_df.to_csv(basic_path)