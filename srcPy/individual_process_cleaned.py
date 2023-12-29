import os
import subprocess
import zipfile
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import plotly.express as px
from plotly import graph_objs as go
from shapely import wkt
import matplotlib as mpl
from statsmodels.stats.oneway import anova_oneway
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


depto_mapping = {
    'Amazonas': 'Amazonas',
    'Antioquia': 'Antioquia',
    'Arauca': 'Arauca',
    'San Andres': 'Archipiélago de San Andrés, Providencia y Santa Catalina',
    'Atlantico': 'Atlántico',
    'Bogota': 'Bogotá, D.C.',
    'Bogotá': 'Bogotá, D.C.',
    'Bolivar': 'Bolívar',
    'Boyaca': 'Boyacá',
    'Caldas': 'Caldas',
    'Caqueta': 'Caquetá',
    'Casanare': 'Casanare',
    'Cauca': 'Cauca',
    'Cesar': 'Cesar',
    'Choco': 'Chocó',
    'Cordoba': 'Córdoba',
    'Cundinamarca': 'Cundinamarca',
    'Guainia': 'Guainía',
    'Guaviare': 'Guaviare',
    'Huila': 'Huila',
    'La Guajira': 'La Guajira',
    'Magdalena': 'Magdalena',
    'Meta': 'Meta',
    'Nariño': 'Nariño',
    'Norte Santander': 'Norte de Santander',
    'Putumayo': 'Putumayo',
    'Quindio': 'Quindio',
    'Risaralda': 'Risaralda',
    'Santander': 'Santander',
    'Sucre': 'Sucre',
    'Tolima': 'Tolima',
    'Valle': 'Valle del Cauca',
    'Vaupes': 'Vaupés',
    'Vichada': 'Vichada'
}

# Set env variables and folder paths
root = os.environ['ROOT']
zip_folder_path = root + 'raw/zip'
txt_folder_path = root + 'raw/txt'
shapefile_abs_path = root + 'raw/shapefile/col_admbnda_adm1_mgn_20200416.shp'
saber11_folder_path = root + 'processed/saber11'
geoportal_basic_folder_path = root + 'process/geoportal_basic'
geoportal_period_folder_path = root + 'process/geoportal_basic/period'
shapefile_merge_folder_path = root + 'processed/shapefile_merge'
geoportal_merge_folder_path = root + 'processed/geoportal_merge'
heatmap_folder_path = root + 'plot/heatmap'
choropleth_folder_path = root + 'plot/choropleth'
scatter_folder_path = root + 'plot/scatter'
anova_table_path = root+ 'table/anova'
linear_reg_path = root+ 'table/linear_regression'
zip_files = [file for file in os.listdir(zip_folder_path) if file.endswith(".zip")]
os.makedirs(zip_folder_path, exist_ok=True)
os.makedirs(txt_folder_path, exist_ok=True)
os.makedirs(saber11_folder_path, exist_ok=True)
os.makedirs(shapefile_merge_folder_path, exist_ok=True)
os.makedirs(heatmap_folder_path, exist_ok=True)

# Open zip file
for zip_file in zip_files:
    print(zip_file, ' File unzipping...')
    zip_file_path = os.path.join(zip_folder_path, zip_file)
    
    # Check if the text file has been extracted
    txt_file_name = zip_file.replace('.zip', '.txt')
    if txt_file_name not in os.listdir(txt_folder_path):
        # 'r' mode to read an existing file
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            # Extract everything in zip to the path
            zip_ref.extractall(txt_folder_path)

# Clean txt file
for txt_file in os.listdir(txt_folder_path):
    if txt_file.endswith(".TXT") or txt_file.endswith(".txt"):
        print(txt_file, ' File cleaning...')
        input_txt_file = os.path.join(txt_folder_path, txt_file)
        output_csv_file = os.path.join(saber11_folder_path, txt_file.replace('.txt', '_tr.csv').replace('.TXT', '_tr.csv'))

        # Check if the processed csv file exists
        if not os.path.exists(output_csv_file):
            subprocess.run(["tr", "¬¨", ";"], input=open(input_txt_file, "rb").read(), stdout=open(output_csv_file, "wb"))

shapefile = gpd.read_file(shapefile_abs_path, crs='EPSG:4326')
combined_data = pd.DataFrame()
distance = dict()
count = 0

# Process csv file
for csv_file in [x for x in os.listdir(saber11_folder_path) if x.endswith('_tr.csv')]:
    csv_file_path = os.path.join(saber11_folder_path, csv_file)
    df = pd.read_csv(csv_file_path, sep=';', encoding='utf-8', low_memory=False)
    print(csv_file, df.shape)

    # Clean missing value of PUNT_ attributes
    df.columns = df.columns.str.upper()
    filtered_columns = df.columns[df.columns.str.match(r'^(P|p)(U|u)(N|n)(T_|t_)')]
    # exclude_column = ['PUNT_RAZONA_CUANTITATIVO', 'PUNT_COMP_CIUDADANA']
    # filtered_columns = filtered_columns.drop(exclude_column)
    filtered_missing_df = df.dropna(subset=filtered_columns)

    # Combine each period together
    # combined_data = pd.concat([combined_data, filtered_missing_df])

    # Calculate correlation, Plot heatmap
    heatmap_image_file = os.path.join(heatmap_folder_path, csv_file.replace(".csv", "_heatmap.pdf"))
    if not os.path.exists(heatmap_image_file):
        selected_columns = filtered_missing_df.filter(regex=r'^(P|p)(U|u)(N|n)(T_|t_)', axis=1)
        correlation_pd = selected_columns.corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_pd, annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Correlation Matrix")
        plt.xticks(fontsize=6, rotation=45)
        plt.yticks(fontsize=6)
        plt.savefig(heatmap_image_file, dpi=300, format='pdf')
        plt.clf()

    merged_data_file = os.path.join(shapefile_merge_folder_path, csv_file.replace(".csv", "_shape_merged.csv"))
    if not os.path.exists(merged_data_file):
        # Merge csv file with shape file: Add 'State' Column and other shape attributes
        filtered_missing_df['ESTU_DEPTO_RESIDE'] = filtered_missing_df['ESTU_DEPTO_RESIDE'].str.title()
        filtered_missing_df['ESTU_SHP_DEPTO'] = None
        filtered_missing_df['ESTU_SHP_DEPTO'] = filtered_missing_df['ESTU_DEPTO_RESIDE'].apply(lambda x: depto_mapping[x] if x in depto_mapping else None)
        dissolved_data = shapefile.dissolve(by='ADM1_ES', as_index=False)
        grouped_data = filtered_missing_df.groupby('ESTU_SHP_DEPTO')['PUNT_GLOBAL'].agg(['mean', 'std', lambda x: x.sem()]).reset_index()
        grouped_data = grouped_data.rename(columns={'<lambda_0>': 'SEM'})
        merged_data = dissolved_data.merge(grouped_data, how='left', left_on='ADM1_ES', right_on='ESTU_SHP_DEPTO').fillna(0)

        # Add 'Distance to capital' Column
        centroids = merged_data.geometry.centroid
        centroids_df = pd.DataFrame({'ADM1_ES': merged_data['ADM1_ES'], 'Latitude': centroids.y, 'Longitude': centroids.x})
        gdf = gpd.GeoDataFrame(centroids_df, geometry=gpd.points_from_xy(centroids_df.Longitude, centroids_df.Latitude), crs="EPSG:4326")
        centroids_df = gdf.to_crs('EPSG:6247')
        capital = centroids_df[centroids_df['ADM1_ES'] == 'Bogotá, D.C.']
        point_end = Point(capital.geometry.x, capital.geometry.y)
        merged_data.columns = merged_data.columns.str.upper()
        
        if len(distance) != len(centroids_df['ADM1_ES'].unique()):
            for index, row in centroids_df.iterrows():
                point_start = Point(row.geometry.x, row.geometry.y)
                distance[row.ADM1_ES] = point_start.distance(point_end)     
        merged_data['DIST_TO_CAPITAL'] = merged_data['ADM1_ES'].apply(lambda x: distance[x] * 0.00062137 if x in distance else None)
        merged_data.to_csv(merged_data_file, index=False)
    
    merged_data_plot = pd.read_csv(merged_data_file, sep=',', encoding='utf-8', low_memory=False)

    anova_table_file = os.path.join(anova_table_path, csv_file.replace(".csv", "_anova_table.csv"))
    if not os.path.exists(anova_table_file):
        # ANOVA
        depto_grouped = filtered_missing_df[['PUNT_GLOBAL','ESTU_DEPTO_RESIDE']].copy()
        data = depto_grouped[['PUNT_GLOBAL', 'ESTU_DEPTO_RESIDE']]
        model = ols('PUNT_GLOBAL ~ ESTU_DEPTO_RESIDE', data=data).fit()
        anova_table = anova_lm(model)
        df_anova_results = pd.DataFrame(anova_table)
        df_anova_results.to_csv(anova_table_file)

    # Plot choropleth map
    merged_data_plot['GEOMETRY'] = merged_data_plot['GEOMETRY'].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(merged_data_plot,geometry='GEOMETRY', crs='epsg:4326')
    merged_data_plot = gdf.to_crs('EPSG:6247')

    merged_data_map = os.path.join(choropleth_folder_path, csv_file.replace(".csv", "_map.pdf"))
    if not os.path.exists(merged_data_map):
        merged_data_plot.plot(column='MEAN', legend=True) #, legend_kwds={"label": "Mean Global Score", "orientation": "horizontal"})
        plt.title('Mean Global Score in Each State')
        for x, y, punt_global, adm1_es in zip(merged_data_plot.geometry.centroid.x, merged_data_plot.geometry.centroid.y, merged_data_plot['MEAN'], merged_data_plot['ADM1_ES']):
            label = f'{round(punt_global, 1)}\n {adm1_es}'
            plt.annotate(
                text=label,
                xy=(x, y),
                xytext=(3, 3),
                textcoords='offset points',
                ha='center',
                va='center',
                fontsize=3,
                color='black'
            )
        plt.savefig(merged_data_map, dpi=500, format='pdf')
        plt.clf()

    # Plot state/global_score scatter
    merged_data_scatter = os.path.join(scatter_folder_path, csv_file.replace(".csv", "_scatter.pdf"))
    merged_data_plot = merged_data_plot[merged_data_plot['MEAN'] != 0]
    if not os.path.exists(merged_data_scatter):
        fig = px.scatter(merged_data_plot, 
                         x='DIST_TO_CAPITAL',
                         y='MEAN',
                         error_y='SEM',
                         trendline='ols')
        fig.update_traces(textposition='top center')
        fig.update_layout(
            title='Global Score vs Distance to Capital in Miles',
            xaxis_title='Distance To Capital',
            yaxis_title='Global Score',
            font=dict(size=8)
        )
        fig.show()
        fig.write_image(merged_data_scatter, engine="kaleido")

    linear_regression_file = os.path.join(linear_reg_path, csv_file.replace(".csv", "_linear_reg.csv"))
    if not os.path.exists(linear_regression_file):
        X = merged_data_plot[['DIST_TO_CAPITAL']].values.reshape(-1, 1)
        y = merged_data_plot['MEAN']
        model = LinearRegression().fit(X, y)
        results = {
            "coefficients": model.coef_,
            "intercept": model.intercept_,
            "rank": model.rank_,
            "singular_values": model.singular_,
            "n_features": model.n_features_in_
        }
        results_df = pd.DataFrame(results)
        results_df.to_csv(linear_regression_file)


# Combine Global_Punt with geoportal indicators
all_sociaecon = pd.DataFrame()
merged_basic_file = os.path.join(geoportal_merge_folder_path, "geoportal_all_time.csv")
if not os.path.exists(merged_basic_file):
    for csv_file in [x for x in os.listdir(shapefile_merge_folder_path) if x.endswith('_shape_merged.csv')]:
        basic_df = pd.read_csv((geoportal_basic_folder_path + '/geoportal_basic.csv'), sep=',', encoding='utf-8', low_memory=False, index_col=0)
        csv_file_path = os.path.join(shapefile_merge_folder_path, csv_file)
        shape_df = pd.read_csv(csv_file_path, sep=',', encoding='utf-8', low_memory=False)
        print(shape_df.columns)
        print(basic_df.columns)

        basic_df = basic_df.merge(shape_df[['ADM1_ES', 'MEAN']], on='ADM1_ES', how='left')
        print('This is: ', csv_file)
        print(basic_df.shape)

        basic_df.to_csv(os.path.join(geoportal_period_folder_path, csv_file.replace("_shape_merged.csv", "_basic.csv")))
        all_sociaecon = pd.concat([all_sociaecon, basic_df])
        print('Concate: ', all_sociaecon.columns)
        print(all_sociaecon.shape)

    all_sociaecon.reset_index(inplace=True, drop=True)
    all_sociaecon.to_csv(merged_basic_file)
    print('Done')  