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
import warnings
# Desactivar todos los warnings
warnings.filterwarnings('ignore')

#Función para obtener los nombres de las variables del netCDF
def variables_name_nc(f_path):
	# leer el NetCDF file en modo lectura
	with nc.Dataset(f_path, 'r') as file:
		 # Obtener los nombres de las variables del fichero
		variable_names = list(file.variables.keys())
	return variable_names

# Función para obtener el periodo de tiempo
def get_time_period(start_year, end_year):
	return int(start_year), int(end_year)

#Función para generar las métricas y las figuras
def generate_metrics_and_plots(selected_grids, selected_variable, start_year, end_year):
	print('Selected grids: ' + str(selected_grids))
	print('Selected variable: ' + selected_variable)
	print(f'Selected period: {start_year} - {end_year}')
	
	# Cargar datos de estaciones (archivo de ejemplo 'stations_data.csv')
	print('loading stations CSV data...')
	try:
		stations_data_0 = pd.read_csv('stations_data_' + selected_variable + '.csv')
		stations_data_0['date'] = pd.to_datetime(stations_data_0['date'])
		# Filtrar por el periodo especificado
		stations_data_0 = stations_data_0[(stations_data_0['date'].dt.year >= start_year) & (stations_data_0['date'].dt.year <= end_year)]
		stations_data_0 = stations_data_0.dropna()
		print(stations_data_0)
		
	except FileNotFoundError:
		print(f'Error - file not found: stations_data_{selected_variable}.csv')
		exit()
		
	for grid in selected_grids:
		stations_data = stations_data_0 # se crea un nuevo dataframe que contiene los datos de las estaciones del CSV 
		# Cargar datos de la rejilla (ISIMIP-CHELSA, CHIRTS, CHIRPS, ERA5 y ERA5-land) usando netCDF4
		print(grid)
		print('loading netCDF grid data...')
		
		try:
			grid_file = 'grid_data_' + grid + '_' + selected_variable + '.nc'
			grid_data = nc.Dataset(grid_file)

		except:
			print('Error - file no found: grid_data_' + grid + '_' + selected_variable + '.nc')
			exit()
			          
		# Definir nombres de variables
		names = variables_name_nc(grid_file)
		targetvar = names[-1]
		targetlat = [string for string in names if 'lat' in string][0]
		targetlon = [string for string in names if 'lon' in string][0]
		targettime = [string for string in names if string == 'time'][0]

		# Variables dentro del archivo netCDF y conversión de las unidades
		grid_lat = grid_data.variables[targetlat][:]
		grid_lon = grid_data.variables[targetlon][:]
		grid_time = grid_data.variables[targettime][:]
		units = grid_data.variables[targettime].units
		
		if selected_variable in ['temperature', 'maximum_temperature', 'minimum_temperature'] and grid != 'CHIRTS':
			grid_targetvar = grid_data.variables[targetvar][:].astype('float32') - 273.15 # convierte grados Kelvin a grados Celsius
		elif selected_variable in ['temperature', 'maximum_temperature', 'minimum_temperature'] and grid == 'CHIRTS':
			grid_targetvar = grid_data.variables[targetvar][:].astype('float32') # mantiene las unidades en grados Celsius
		elif selected_variable == 'precipitation' and grid != 'CHIRPS' and grid != 'ISIMIP-CHELSA':
			grid_targetvar = grid_data.variables[targetvar][:].astype('float32')*1000 # convierte m/día a mm/día
		elif selected_variable == 'precipitation' and grid == 'CHIRPS':
			grid_targetvar = grid_data.variables[targetvar][:].astype('float32') # mantiene las unidades en mm/día
		elif selected_variable == 'precipitation' and grid == 'ISIMIP-CHELSA':
			grid_targetvar = grid_data.variables[targetvar][:].astype('float32')*86400 # convierte kg/m²s a mm/día
		else:
			print('Error - units')
			exit()
		grid_data.close() # se cierra el netCDF
		del grid_data

		# Función para convertir fechas a índices de tiempo en la rejilla 
		def convert_time_to_index(time_array, date):
			time_num = nc.date2num(date, units=units, calendar='standard')#'days since 1991-01-01 00:00:00', calendar='standard')
			time_idx = np.interp(time_num, time_array, np.arange(len(time_array)))
			return time_idx

		
		# Crear el interpolador para los datos de la rejilla 
		def create_interpolator(targetvar_data, lat_array, lon_array, lat_station, lon_station):
			lat_array = np.sort(np.unique(lat_array))
			lon_array = np.sort(np.unique(lon_array))		
			lat_idx = np.abs(lat_array - lat_station).argmin()
			lon_idx = np.abs(lon_array - lon_station).argmin()

			# Definir rangos para interpolación local
			lat_range = lat_array[max(0, lat_idx-1):min(len(lat_array), lat_idx+2)]
			lon_range = lon_array[max(0, lon_idx-1):min(len(lon_array), lon_idx+2)]
			targetvar_range = targetvar_data[:, max(0, lat_idx-1):min(len(lat_array), lat_idx+2), max(0, lon_idx-1):min(len(lon_array), lon_idx+2)]
			
			return RegularGridInterpolator(
				(np.arange(len(grid_time)), lat_range, lon_range), 
				targetvar_range,
				method='nearest',
				bounds_error=False,
				fill_value=np.nan
			)

		# Función para extraer valores interpolados de la rejilla para las ubicaciones y fechas de las estaciones 
		def extract_interpolated_grid_value(lat, lon, date):
			time_idx = convert_time_to_index(grid_time, date)
			interpolator = create_interpolator(grid_targetvar, grid_lat, grid_lon, lat, lon)
			interpolated_value = interpolator((time_idx, lat, lon))
			return interpolated_value
			
		print('data loaded')
		
		# Aplica la extracción a cada fila del DataFrame de estaciones usando la interpolación local
		print('interpolating...')
		print('please wait')
		stations_data['interpolated_grid_value'] = stations_data.apply(
			lambda row: extract_interpolated_grid_value(row['latitude'], row['longitude'], row['date']),
			axis=1
		)
		
		# Calcular diferencias y métricas
		stations_data['interpolated_grid_value'] = stations_data['interpolated_grid_value'].apply(lambda x: x.filled(np.nan) if isinstance(x, np.ma.MaskedArray) else x) # los valores enmascarados se convierten a NaN
		stations_data = stations_data.dropna() # se eliminan filas con NaN
		#print(stations_data[~stations_data['interpolated_grid_value'].apply(lambda x: isinstance(x, (int, float)))])
		#print(stations_data[~stations_data[selected_variable].apply(lambda x: isinstance(x, (int, float)))])
		stations_data['interpolated_grid_value'] = pd.to_numeric(stations_data['interpolated_grid_value'], errors='coerce') # todos los valores se convierten a formato numérico
		stations_data[selected_variable] = pd.to_numeric(stations_data[selected_variable], errors='coerce')
		print('interpolation completed')
		print('obtaining metrics...')
		print('please wait')
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
			metrics_per_station_interpolated['Number of wet days Bias'] = metrics_per_station_accumulated['Number of wet days Bias'] / (stations_data['date'].nunique() / 365)

			metrics_per_station_interpolated.to_csv('metrics_per_station_interpolated_' + grid + '_' + selected_variable + '.csv', index=False)
			print('metrics_per_station_interpolated_' + grid + '_' + selected_variable + '.csv has been saved')
			
		else:
			print(f'Error: variable {selected_variable} no contemplada para el cálculo de métricas.')
			continue

		# Calcular el ciclo anual
		
		dfac = stations_data.dropna()
		dfac['month'] = dfac['date'].dt.month
		dfac['interpolated_grid_value'] = dfac['interpolated_grid_value'].astype(float)
		# Agrupar por mes y calcular el ciclo anual promediando con todas las estaciones observacionales
		if selected_variable != 'precipitation':
			monthly_avg = dfac.groupby('month').agg({
				selected_variable: 'mean',
				'interpolated_grid_value': 'mean'
			}).reset_index()
		else:
			monthly_avg = dfac.groupby('month').agg({
				selected_variable: 'sum',
				'interpolated_grid_value': 'sum'
			}).reset_index()
			
			columns_to_divide = [selected_variable, 'interpolated_grid_value']
			monthly_avg[columns_to_divide] = monthly_avg[columns_to_divide] / (stations_data['station_id'].nunique() * stations_data['date'].nunique() / 365)
		monthly_avg.to_csv(selected_variable + '_' + grid + '_annual_cycle_comparison.csv', index=False)
		
		del stations_data
		del stations_data_accu_ini
		del dfac
		
	# Diccionario para almacenar los datos de métricas
	metrics_data_dict = {}
	# Diccionario para almacenar los datos del ciclo anual
	annual_cycle_dict = {}

	# Cargar las métricas de cada grid desde los CSV
	for grid in selected_grids:  
		file_name = f'metrics_per_station_interpolated_{grid}_{selected_variable}.csv'
		file_name_annual_cycle = selected_variable + '_' + grid + '_annual_cycle_comparison.csv'
		try:
			metrics_data = pd.read_csv(file_name)
			metrics_data_dict[grid] = metrics_data
			
			annual_cycle_data = pd.read_csv(file_name_annual_cycle)
			annual_cycle_dict[grid] = annual_cycle_data 
			
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
 
	# Graficar el ciclo anual
	plt.figure(figsize=(12, 6))
	plt.plot(annual_cycle_dict[grid]['month'], annual_cycle_dict[grid][selected_variable], label= 'observations', marker='o')
	for grid in selected_grids:
		plt.plot(annual_cycle_dict[grid]['month'], annual_cycle_dict[grid]['interpolated_grid_value'], label= grid, marker='o')
	# Configurar el gráfico
	plt.xlabel('Month')
	if selected_variable != 'precipitation':
		plt.ylabel(selected_variable + ' (°C)')
	else:
		plt.ylabel(selected_variable + ' (mm)')
	plt.title('Average Annual Cycle')
	# Etiquetas de los meses abreviados en inglés
	months_abbr = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
	# Usar plt.xticks() para asignar etiquetas de texto a los meses
	plt.xticks(annual_cycle_dict[grid]['month'], months_abbr)
	plt.legend()
	plt.savefig(f'{selected_variable}_annual_cycle_grids_comparison.png')
	print(f'{selected_variable}_annual_cycle_grids_comparison.png has been saved')
	
	
def on_generate_button_click():
	selected_variables = [combo_variable.get()]
	selected_grids = listbox_grids.curselection()
	selected_grids = [grids[i] for i in selected_grids]
	start_year = entry_start_year.get()
	end_year = entry_end_year.get()

	# Validar entradas de año
	if not start_year.isdigit() or not end_year.isdigit():
		messagebox.showerror("Error", "Por favor ingresa años válidos.")
		return
	if int(start_year) > int(end_year):
		messagebox.showerror("Error", "El año de inicio debe ser menor o igual al año de fin.")
		return

	generate_metrics_and_plots(selected_grids, selected_variables[0], int(start_year), int(end_year))

# Variables y lista de rejillas
variables = ['temperature', 'maximum_temperature', 'minimum_temperature', 'precipitation']
grids = ['ISIMIP-CHELSA', 'CHIRTS', 'CHIRPS', 'ERA5', 'ERA5-Land']

# Crear la GUI
root = tk.Tk()
root.title('Grid Evaluator')

# Etiquetas y ComboBox para seleccionar la variable
label_variable = ttk.Label(root, text='Select Variable:')
label_variable.pack(pady=10)
combo_variable = ttk.Combobox(root, values=variables)
combo_variable.pack()

# Etiqueta y Listbox para seleccionar las rejillas
label_grids = ttk.Label(root, text='Select Grids:')
label_grids.pack(pady=10)
listbox_grids = tk.Listbox(root, selectmode=tk.MULTIPLE, exportselection=0)
for grid in grids:
    listbox_grids.insert(tk.END, grid)
listbox_grids.pack()

# Etiquetas y entradas para seleccionar el periodo de tiempo
label_period = ttk.Label(root, text='Select Period:')
label_period.pack(pady=10)

frame_period = ttk.Frame(root)
frame_period.pack(pady=10)

label_start_year = ttk.Label(frame_period, text='Start Year:')
label_start_year.grid(row=0, column=0, padx=5)

entry_start_year = ttk.Entry(frame_period, width=10)
entry_start_year.grid(row=0, column=1, padx=5)
entry_start_year.insert(0, '1991')  # Año de inicio por defecto

label_end_year = ttk.Label(frame_period, text='End Year:')
label_end_year.grid(row=0, column=2, padx=5)

entry_end_year = ttk.Entry(frame_period, width=10)
entry_end_year.grid(row=0, column=3, padx=5)
entry_end_year.insert(0, '2020')  # Año de fin por defecto

# Botón para generar las métricas y gráficos
generate_button = ttk.Button(root, text='Generate Metrics & Plots', command=on_generate_button_click)
generate_button.pack(pady=20)

# Iniciar la interfaz gráfica
root.mainloop()
