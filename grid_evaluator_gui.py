import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import netCDF4 as nc
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import pearsonr, spearmanr


def generate_metrics_and_plots(selected_grids, selected_variable):
	print('Selected grids: ' + str(selected_grids))
	print('Selected variable: ' + selected_variable)

	# Crear un DataFrame vacío para almacenar las métricas de todos los grids
	all_metrics = pd.DataFrame()

	for grid in selected_grids:
		# Cargar datos de la rejilla (ISIMIP-CHELSA, CHIRTS, CHIRPS, ERA5 y ERA5-land) usando netCDF4
		try:
			grid_file = 'grid_data_' + grid + '_' + selected_variable + '.nc'	
			grid_data = nc.Dataset(grid_file)
		except:
			print('Error - file no found: grid_data_' + grid + '_' + selected_variable + '.nc')
			exit()
		# Cargar datos de estaciones (archivo de ejemplo 'stations_data.csv')
		try:
			stations_data = pd.read_csv('stations_data_' + selected_variable + '.csv')
			stations_data['date'] = pd.to_datetime(stations_data['date'])
		except:
			print('Error - file no found: stations_data_' + selected_variable + '.csv')
			exit()
		# Definir nombres de variables
		if grid == 'ISIMIP-CHELSA':
			if selected_variable == 'temperature':
				targetvar = 'tas'
				targetlat = 'lat'
				targetlon = 'lon'
				targettime = 'time'
			elif selected_variable == 'maximum_temperature':
				targetvar = 'tasmax'
				targetlat = 'lat'
				targetlon = 'lon'
				targettime = 'time'
			elif selected_variable == 'minimum_temperature':
				targetvar = 'tasmin'
				targetlat = 'lat'
				targetlon = 'lon'
				targettime = 'time'
			elif selected_variable == 'precipitation':
				targetvar = 'pr'
				targetlat = 'lat'
				targetlon = 'lon'
				targettime = 'time'
			else:
				print(f'Error: {selected_variable} not found in grid {grid}')
				continue

		elif grid == 'CHIRTS':
			if selected_variable == 'maximum_temperature':
				targetvar = 'Tmax'
				targetlat = 'latitude'
				targetlon = 'longitude'
				targettime = 'time'
			elif selected_variable == 'minimum_temperature':
				targetvar = 'Tmin'
				targetlat = 'latitude'
				targetlon = 'longitude'
				targettime = 'time'
			else:
				print(f'Error: {selected_variable} not found in grid {grid}')
				continue

		elif grid == 'CHIRPS':
			if selected_variable == 'precipitation':
				targetvar = 'precip'
				targetlat = 'latitude'
				targetlon = 'longitude'
				targettime = 'time'
			else:
				print(f'Error: {selected_variable} not found in grid {grid}')
				continue

		elif grid == 'ERA5':
			if selected_variable == 'temperature':
				targetvar = '2t'
				targetlat = 'lat'
				targetlon = 'lon'
				targettime = 'time'
			elif selected_variable == 'maximum_temperature':
				targetvar = 'mx2t'
				targetlat = 'lat'
				targetlon = 'lon'
				targettime = 'time'
			elif selected_variable == 'minimum_temperature':
				targetvar = 'mn2t'
				targetlat = 'lat'
				targetlon = 'lon'
				targettime = 'time'
			elif selected_variable == 'precipitation':
				targetvar = 'tp'
				targetlat = 'lat'
				targetlon = 'lon'
				targettime = 'time'
			else:
				print(f'Error: {selected_variable} not found in grid {grid}')
				continue

		elif grid == 'ERA5-Land':
			if selected_variable == 'temperature':
				targetvar = '2t'
				targetlat = 'lat'
				targetlon = 'lon'
				targettime = 'time'
			elif selected_variable == 'maximum_temperature':
				targetvar = 'mx2t'
				targetlat = 'lat'
				targetlon = 'lon'
				targettime = 'time'
			elif selected_variable == 'minimum_temperature':
				targetvar = 'mn2t'
				targetlat = 'lat'
				targetlon = 'lon'
				targettime = 'time'
			elif selected_variable == 'precipitation':
				targetvar = 'tp'
				targetlat = 'lat'
				targetlon = 'lon'
				targettime = 'time'
			else:
				print(f'Error: {selected_variable} not found in grid {grid}')
				continue

		else:
			if selected_variable == 'temperature':
				targetvar = 'temperature'
				targetlat = 'lat'
				targetlon = 'lon'
				targettime = 'time'
			elif selected_variable == 'precipitation':
				targetvar = 'precipitation'
				targetlat = 'lat'
				targetlon = 'lon'
				targettime = 'time'
			else:
				print(f'Error: {selected_variable} not found in grid {grid}')
				continue
		
		# Variables dentro del archivo netCDF (los nombres habrá que adaptarlos en función de la rejilla, en este ejemplo la variable climática es temperatura)
		grid_lat = grid_data.variables[targetlat][:]
		grid_lon = grid_data.variables[targetlon][:]
		grid_time = grid_data.variables[targettime][:]
		grid_targetvar = grid_data.variables[targetvar][:]

		# Función para convertir fechas a índices de tiempo en la rejilla 
		def convert_time_to_index(time_array, date):
			time_num = nc.date2num(date, units='days since 1991-01-01 00:00:00', calendar='standard')
			time_idx = np.interp(time_num, time_array, np.arange(len(time_array)))
			return time_idx

		# Crear el interpolador para los datos de la rejilla (solo local alrededor de la estación para ganar eficiencia computacional)
		def create_interpolator(targetvar_data, lat_array, lon_array, lat_station, lon_station):
			lat_idx = np.abs(lat_array - lat_station).argmin()
			lon_idx = np.abs(lon_array - lon_station).argmin()
			# Definir rangos para interpolación local
			lat_range = lat_array[max(0, lat_idx-1):min(len(lat_array), lat_idx+2)]
			lon_range = lon_array[max(0, lon_idx-1):min(len(lon_array), lon_idx+2)]
			targetvar_range = targetvar_data[:, max(0, lat_idx-1):min(len(lat_array), lat_idx+2), max(0, lon_idx-1):min(len(lon_array), lon_idx+2)]
			
			return RegularGridInterpolator(
				(np.arange(len(grid_time)), lat_range, lon_range), 
				targetvar_range,
				bounds_error=False,
				fill_value=None
			)

		# Función para extraer valores interpolados de la rejilla para las ubicaciones y fechas de las estaciones
		def extract_interpolated_grid_value(lat, lon, date):
			interpolator = create_interpolator(grid_targetvar, grid_lat, grid_lon, lat, lon)
			time_idx = convert_time_to_index(grid_time, date)
			return interpolator((time_idx, lat, lon))
			
		
		# Función para extraer valores interpolados de la rejilla para las ubicaciones y fechas de las estaciones
		def extract_interpolated_grid_value_precip(lat, lon, date):
			interpolator = create_interpolator(grid_targetvar, grid_lat, grid_lon, lat, lon)
			time_idx = convert_time_to_index(grid_time, date)
			interpolated_value = interpolator((time_idx, lat, lon))
			# Asegurar que el valor interpolado de precipitación no sea negativo
			return max(0, interpolated_value)

		# Aplica la extracción a cada fila del DataFrame de estaciones usando la interpolación local
		
		if selected_variable in ['temperature', 'maximum_temperature', 'minimum_temperature']:
			stations_data['interpolated_grid_value'] = stations_data.apply(
				lambda row: extract_interpolated_grid_value(row['latitude'], row['longitude'], row['date']),
				axis=1
			)
		else: 
			stations_data['interpolated_grid_value'] = stations_data.apply(
				lambda row: extract_interpolated_grid_value_precip(row['latitude'], row['longitude'], row['date']),
				axis=1
			)

		# Calcular diferencias y métricas
		stations_data_accu_ini = stations_data
		stations_data['difference_interpolated'] = stations_data['interpolated_grid_value'] - stations_data[selected_variable] 

		def calculate_metrics_interpolated_temp(data):
			data = data.dropna()  # Eliminar filas con NaN si es necesario
			if len(data) < 2:
				return pd.Series({
					'Mean Bias': np.nan,
					'Mean Absolute Error': np.nan,
					'RMSE': np.nan,
					'Correlation': np.nan,
					'Variance Bias': np.nan
				})
			mean_bias = data['difference_interpolated'].mean()
			mean_absolute_error = data['difference_interpolated'].abs().mean()
			rmse = np.sqrt((data['difference_interpolated'] ** 2).mean())
			correlation, _ = pearsonr(data['interpolated_grid_value'], data[selected_variable])
			variance_bias = data['interpolated_grid_value'].var() - data[selected_variable].var()
			percentile90_bias = np.percentile(data['interpolated_grid_value'], 90) - np.percentile(data[selected_variable], 90)
			percentile10_bias = np.percentile(data['interpolated_grid_value'], 10) - np.percentile(data[selected_variable], 10)
			
			return pd.Series({
				'Mean Bias': mean_bias,
				'P90 Bias': percentile90_bias,
				'P10 Bias': percentile10_bias,
				'Mean Absolute Error': mean_absolute_error,
				'RMSE': rmse,
				'Correlation': correlation,
				'Variance Bias': variance_bias
			})

		def calculate_metrics_interpolated_precip(data):
			data = data.dropna()  # Eliminar filas con NaN si es necesario
			if len(data) < 2:
				return pd.Series({
					'Mean Bias': np.nan,
					'Mean Absolute Error': np.nan,
					'Correlation': np.nan,
					'Variance Bias': np.nan
				})
			mean_bias = data['difference_interpolated'].mean()
			mean_absolute_error = data['difference_interpolated'].abs().mean()
			correlation, _ = spearmanr(data['interpolated_grid_value'], data['precipitation'])
			variance_bias = data['precipitation'].var() - data['interpolated_grid_value'].var()
			percentile90_bias = np.percentile(data['interpolated_grid_value'], 90) - np.percentile(data['precipitation'], 90)
			percentile10_bias = np.percentile(data['interpolated_grid_value'], 10) - np.percentile(data['precipitation'], 10)
			percentile95_bias = np.percentile(data['interpolated_grid_value'], 95) - np.percentile(data['precipitation'], 95)
			
			return pd.Series({
				'Mean Bias': mean_bias,
				'P95 Bias': percentile95_bias,
				'P90 Bias': percentile90_bias,
				'P10 Bias': percentile10_bias,
				'Mean Absolute Error': mean_absolute_error,
				'Correlation': correlation,
				'Variance Bias': variance_bias
			})

		# Calcular métricas basadas en la variable seleccionada
		if selected_variable in ['temperature', 'maximum_temperature', 'minimum_temperature']:
			metrics_per_station_interpolated = stations_data.groupby('station_id').apply(calculate_metrics_interpolated_temp).reset_index()
			# Guardar métricas para cada estación en un csv
			metrics_per_station_interpolated.to_csv('metrics_per_station_interpolated_' + grid + '_' + selected_variable + '.csv', index=False)
			print('metrics_per_station_interpolated_' + grid + '_' + selected_variable + '.csv has been saved')
			# Unidades para cada métrica
			units = {
				'Mean Bias': '°C',
				'P90 Bias': '°C',
				'P10 Bias': '°C',
				'Mean Absolute Error': '°C',
				'RMSE': '°C',
				'Correlation': 'Dimensionless',
				'Variance Bias': '°C²'
			}
			
		elif selected_variable == 'precipitation':
			metrics_per_station_interpolated = stations_data.groupby('station_id').apply(calculate_metrics_interpolated_precip).reset_index()
			# Guardar métricas para cada estación en un csv
				
			units = {
				'Mean Bias': 'mm',
				'P95 Bias': 'mm',
				'P90 Bias': 'mm',
				'P10 Bias': 'mm',
				'Mean Absolute Error': 'mm',
				'Correlation': 'Dimensionless',
				'Variance Bias': 'mm²',
				'Number of wet days Bias': 'days'
			}
			
			# Calcular el número de días con precipitación observada mayor a 1 mm
			stations_data_accu_ini['days_with_precipitation'] = stations_data_accu_ini['precipitation'] > 1
			stations_data_accu_ini['days_with_precipitation_interpolated'] = stations_data_accu_ini['interpolated_grid_value'] > 1

			# Agrupar datos por estación y calcular el acumulado de precipitación tanto para datos observados como para datos interpolados
			accumulated_precipitation = stations_data_accu_ini.groupby('station_id').agg({
				'precipitation': 'sum',
				'interpolated_grid_value': 'sum',
				'days_with_precipitation': 'sum',
				'days_with_precipitation_interpolated': 'sum'
			}).reset_index()

			# Calcular diferencias y métricas basadas en el acumulado de precipitación
			accumulated_precipitation['difference_days'] =  accumulated_precipitation['days_with_precipitation_interpolated'] - accumulated_precipitation['days_with_precipitation']

			def calculate_metrics_accumulated(data):
				R01_bias = data['difference_days'].mean()
				return pd.Series({
					'Number of wet days Bias': R01_bias
				})

			metrics_per_station_accumulated = accumulated_precipitation.groupby('station_id').apply(calculate_metrics_accumulated).reset_index()
			
			# Guardar métricas para cada estación en un csv
			metrics_per_station_interpolated['Number of wet days Bias'] = metrics_per_station_accumulated['Number of wet days Bias']

			metrics_per_station_interpolated.to_csv('metrics_per_station_interpolated_' + grid + '_' + selected_variable + '.csv', index=False)
			print('metrics_per_station_interpolated_' + grid + '_' + selected_variable + '.csv has been saved')
			
		else:
			print(f'Error: variable {selected_variable} no contemplada para el cálculo de métricas.')
			continue



	# Crear un DataFrame vacío para almacenar todas las métricas
	all_metrics = pd.DataFrame()

	# Diccionario para almacenar los datos de métricas
	metrics_data_dict = {}

	# Cargar las métricas de cada grid desde los CSV
	for grid in selected_grids:  
		file_name = f'metrics_per_station_interpolated_{grid}_{selected_variable}.csv'
		try:
			metrics_data = pd.read_csv(file_name)
			metrics_data_dict[grid] = metrics_data
		except FileNotFoundError:
			print(f'Warning: No metrics file found for {grid} and {selected_variable}')

	# Verificar los datos cargados
	for grid, metrics_data in metrics_data_dict.items():
		print(f'Loaded metrics data for {grid} and {selected_variable}')
		# print(metrics_data)
		#print(metrics_data.head())

	# Crear una lista de dataframes para pasar a seaborn
	dfs = list(metrics_data_dict.values())

	# Concatenar los dataframes
	metrics_concat = pd.concat(dfs, keys=metrics_data_dict.keys(), names=['Grid'])

	# Resetear completamente el índice para asegurarnos de que sea simple
	metrics_concat = metrics_concat.reset_index()

	# Eliminar la columna 'level_1' si existe
	if 'level_1' in metrics_concat.columns:
		metrics_concat = metrics_concat.drop('level_1', axis=1)

	# Obtener automáticamente la lista de métricas disponibles
	metrics_to_plot = metrics_concat.drop(['Grid', 'station_id'], axis=1).columns.tolist()

	# Generar boxplots para cada métrica
	for metric in metrics_to_plot:
		plt.figure(figsize=(10, 6))
		sns.boxplot(data=metrics_concat, x='Grid', y=metric, hue=None, orient='v', dodge=False)
		plt.title(f'Comparison of {metric} for {selected_variable}')
		plt.ylabel(f'{metric} ({units.get(metric, "")})')
		plt.xlabel('Grid')
		plt.xticks(rotation=45)
		plt.tight_layout()
		plt.savefig(f'{selected_variable}_{metric}_grids_comparison.png')
		print(f'{selected_variable}_{metric}_grids_comparison.png has been saved')
		plt.close()
		#plt.show()
    
def on_generate_button_click():
    selected_variables = [combo_variable.get()]
    selected_grids = listbox_grids.curselection()
    selected_grids = [grids2[i] for i in selected_grids]
    generate_metrics_and_plots(selected_grids, selected_variables[0])



variables = [
    'temperature',
    'maximum_temperature',
    'minimum_temperature',
    'precipitation',
]

# Lista de grids y variables disponibles

grids2 = [
    'ISIMIP-CHELSA',
    'CHIRTS',
    'CHIRPS',
    'ERA5',
    'ERA5-Land',
]

# Crear la GUI
root = tk.Tk()
root.title('Grid Evaluator ')

# Etiquetas y ComboBox para seleccionar la variable
label_variable = ttk.Label(root, text='Select Variable:')
label_variable.pack(pady=10)
combo_variable = ttk.Combobox(root, values=variables)
combo_variable.pack()

# Etiqueta y Listbox para seleccionar los grids
label_grids = ttk.Label(root, text='Select Grids:')
label_grids.pack(pady=10)
listbox_grids = tk.Listbox(root, selectmode=tk.MULTIPLE, exportselection=0)
for grid in grids2:
    listbox_grids.insert(tk.END, grid)
listbox_grids.pack()

# Botón para generar métricas y gráficos
btn_generate = ttk.Button(root, text='Generate Metrics (csv) and Boxplots (png)', command=on_generate_button_click)
btn_generate.pack(pady=20)

root.mainloop()