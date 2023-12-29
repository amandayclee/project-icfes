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
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
from sklearn.tree import plot_tree
simplefilter("ignore", category=ConvergenceWarning)

# Set env variables and folder paths
root = os.environ['ROOT']
geoportal_period_folder_path = root + 'processed/geoportal_basic/period'
model_output_model_path = root + 'model_output'

selected_cols_for_lasso = [
    'percentage_of_households_with_access_to_garbage_collecting_services_valor',
    'percentage_of_households_with_access_to_electric_energy_services_valor',
    'percentage_of_households_with_access_to_natural_gas_services_valor',
    'percentage_of_households_with_access_to_sewer_services_valor',
    'percentage_of_households_with_access_to_water_services_valor',
    'percentage_of_households_with_access_to_internet_services_(home_or_mobile)_valor',
    'population_projections_total',
    'literacy_of_people_15yrs_of_age+_valor',
    'illeteracy_of_people_15yrs_of_age+_valor',
    'masculity_and_femininity_index_indice_feminidad',
    'masculity_and_femininity_index_indice_masculinidad',
    'dependency_index__indice_depen_demo_60mas',
    'dependency_index__indice_depen_demo_65mas',
    'aging_index_indice_envejecimiento_60mas',
    'aging_index_indice_envejecimiento_65mas',
    'percentage_of_population_older_than_5yrs_old_attending_an_educational_institution_valor',
    'youth_index_indice_juventud',
    'percentage_of_raizal_population_valor',
    'percentage_of_indegenous_population_valor',
    'percentage_of_black._mulato._afrodescendant._afrocolombian_population_valor',
    'percentage_of_palenquero_population_valor',
    'percentage_of_gypsy_or_rrom_population_valor',
    'population_density_per_kilometer_squared_2018_pob_km',
    'housing_deficit_deficit_cualitativo',
    'housing_deficit_deficit_cuantitativo',
    'housing_deficit_deficit_habitacional',
    'quantitative_housing_deficit_tipo_de_vivienda',
    'quantitative_housing_deficit_material_de_paredes',
    'quantitative_housing_deficit_cohabitacion',
    'quantitative_housing_deficit_haci_no_miti',
    'unmet_basic_needs_(nbi)_per_category_%_hacinamiento',
    'unmet_basic_needs_(nbi)_per_category_%_inasistencia',
    'unmet_basic_needs_(nbi)_per_category_%_miseria',
    'unmet_basic_needs_(nbi)_per_category_%_personas_nbi',
    'unmet_basic_needs_(nbi)_per_category_%_servicios',
    'unmet_basic_needs_(nbi)_per_category_%_vivienda',
    'qualitative_housing_deficit_haci_miti',
    'qualitative_housing_deficit_material_de_pisos',
    'qualitative_housing_deficit_cocina',
    'qualitative_housing_deficit_agua_para_cocinar',
    'qualitative_housing_deficit_alcantarillado',
    'qualitative_housing_deficit_energia',
    'qualitative_housing_deficit_recoleccion_de_basuras',
    'multidimensional_poverty_index_analfabetismo',
    'multidimensional_poverty_index_ba_ac_se_sa',
    'multidimensional_poverty_index_baj_log_edu',
    'multidimensional_poverty_index_bsp_cpi',
    'multidimensional_poverty_index_hacinamiento_critico',
    'multidimensional_poverty_index_ina_eli_exc',
    'multidimensional_poverty_index_inasistencia_escolar',
    'multidimensional_poverty_index_ipm',
    'multidimensional_poverty_index_ma_in_pa_ex',
    'multidimensional_poverty_index_mat_ina_pis',
    'multidimensional_poverty_index_rezago_escolar',
    'multidimensional_poverty_index_si_ac_fu_ag',
    'multidimensional_poverty_index_sin_ale_sal',
    'multidimensional_poverty_index_trabajo_infantil',
    'infant_mortality_rate_tasa'
]


basic_df = pd.read_csv(geoportal_period_folder_path + '/geoportal_all_time.csv', index_col=0)
basic_df = basic_df[(basic_df['MEAN'] != 0) & basic_df['MEAN'].notna()]
basic_df = basic_df[basic_df['ADM1_ES'] != 'Ninguno'].dropna()

X = basic_df[selected_cols_for_lasso]
Y = basic_df['MEAN']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
print('training set size: {}; test set size: {}'.format(X_train.shape[0], X_test.shape[0]))

alphas = [ x/10**5 for x in range(1, 10**5, 100) ]
model = LassoCV(cv=5, alphas = alphas, random_state=0, max_iter=20000)
model.fit(X_train, Y_train)

print("The best value of penalization chosen by 5-fold cross validation is {}".format(model.alpha_))
lasso_best = Lasso(alpha=model.alpha_)
lasso_best.fit(X_train, Y_train)

vars_LASSO = pd.DataFrame(data = lasso_best.coef_[lasso_best.coef_>0],
     index = X_train.columns[lasso_best.coef_>0],
     columns = ['Coefficients'])

vars_LASSO.to_csv(os.path.join(model_output_model_path, 'LASSO_coefficients.csv'))

selected_lasso_columns = X_train.columns[lasso_best.coef_ > 0].tolist()
print('Attribute numbers before LASSO: ', len(X.columns))
print('Attribute numbers after LASSO: ', len(selected_lasso_columns))

print('R square of LASSO from training set:', round(lasso_best.score(X_train, Y_train)*100, 2))
print('R square of LASSO from test set:', round(lasso_best.score(X_test, Y_test)*100, 2))
print('Root Mean Squared Error of LASSO from training set: {:.4f}'.format(mean_squared_error(Y_train, lasso_best.predict(X_train), squared = False)))
print('Root Mean Squared Error of LASSO from test set: {:.4f}'.format(mean_squared_error(Y_test, lasso_best.predict(X_test), squared = False)))
print('Mean Absolute Error of LASSO from training set: {:.4f}'.format(mean_absolute_error(Y_train, lasso_best.predict(X_train))))
print('Mean Absolute Error of LASSO from test set: {:.4f}'.format(mean_absolute_error(Y_test, lasso_best.predict(X_test))))

# print(selected_cols_for_lasso)
# print(X_train.columns)

X_train, X_test, Y_train, Y_test = X_train[selected_lasso_columns], X_test[selected_lasso_columns], Y_train, Y_test
print('training set size: {}; test set size: {}'.format(X_train.shape[0], X_test.shape[0]))

param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 3, 5, 7, 10]
}

rf = RandomForestRegressor()

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5)
grid_search.fit(X_train, Y_train)

print("Best Parameters:", grid_search.best_params_)
best_rf = grid_search.best_estimator_

print('Root Mean Squared Error of Random Forest from training set: {:.4f}'.format(mean_squared_error(Y_train, best_rf.predict(X_train), squared = False)))
print('Mean Absolute Error of Random Forest from training set: {:.4f}'.format(mean_squared_error(Y_train, best_rf.predict(X_train))))
print('Root Mean Squared Error of Random Forest from testing set: {:.4f}'.format(mean_squared_error(Y_test, best_rf.predict(X_test), squared = False)))
print('Mean Absolute Error of Random Forest from testing set: {:.4f}'.format(mean_squared_error(Y_test, best_rf.predict(X_test))))

feature_importance = best_rf.feature_importances_
feature_names = X_train.columns

feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
top_5_features = feature_importance_df.head(5)
print('Top 5 important features are:', top_5_features)

plt.figure(figsize=(50, 30))
plot_tree(best_rf.estimators_[0], feature_names=X_train.columns, filled=True, rounded=True)
plt.savefig(os.path.join(model_output_model_path, ("rf_training.pdf")), dpi=300, format='pdf')


plt.figure(figsize=(50, 30))
plot_tree(best_rf.estimators_[0], feature_names=X_test.columns, filled=True, rounded=True)
plt.savefig(os.path.join(model_output_model_path, ("rf_testing.pdf")), dpi=300, format='pdf')