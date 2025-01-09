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
import cartopy.crs as ccrs
import cartopy.feature as cfeature
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
def generate_metrics_and_plots(selected_grids, selected_variable, start_year, end_year, selected_months, interpolation_method):
	print('Selected grids: ' + str(selected_grids))
	print('Selected variable: ' + selected_variable)
	print('Selected interpolation method: ' + interpolation_method)
	print(f'Selected period: {start_year} - {end_year}')
	print(f'Selected months: {selected_months}')

	# Cargar datos de estaciones (archivo de ejemplo 'stations_data.csv')
	print('loading stations CSV data...')
	try:
		stations_data_0 = pd.read_csv('stations_data_' + selected_variable + '.csv')
		stations_data_0['date'] = pd.to_datetime(stations_data_0['date'])
		
		# Filtrar por el periodo especificado
		stations_data_0 = stations_data_0[(stations_data_0['date'].dt.year >= start_year) & (stations_data_0['date'].dt.year <= end_year)]
		stations_data_0 = stations_data_0[stations_data_0['date'].dt.month.isin(selected_months)]
		print(stations_data_0)
		
		# Convertir la columna 'date' a tipo datetime
		stations_data_1 = stations_data_0
		stations_data_1['date'] = pd.to_datetime(stations_data_1['date'])
		
		# Crear el rango completo de fechas para el periodo deseado
		start_date = f"{start_year}-01-01"
		end_date = f"{end_year}-12-31"
		date_range = pd.date_range(start=start_date, end=end_date)
		date_range = date_range[date_range.month.isin(selected_months)]
		
		# Agrupar por estación
		stations = stations_data_1['station_id'].unique()
		results = []
		for station_id in stations:
			
			# Filtrar datos de la estación
			df_station = stations_data_1[stations_data_1['station_id'] == station_id]
			
			# Detectar días faltantes en el DataFrame
			station_date_range = date_range
			missing_days = station_date_range.difference(df_station['date'])
			
			# Detectar días con valores faltantes y con valores de relleno específicos
			df_station['is_nan'] = df_station[selected_variable].isna()
			df_station['is_filled'] = df_station[selected_variable].isin([-99, -999, -9999])
			nan_days = df_station[df_station['is_nan']]['date']
			filled_days = df_station[df_station['is_filled']]['date']
			
			# Calcular métricas de completitud
			total_days = len(station_date_range)
			recorded_days = total_days - len(missing_days) - len(nan_days) - len(filled_days)
			completeness_percentage = (recorded_days / total_days) * 100
			
			# Guardar resultados de completitud
			results.append({
				"station_id": station_id,
				"latitude": df_station['latitude'].iloc[0],
				"longitude": df_station['longitude'].iloc[0],
				"total_days": total_days,
				"missing_days_count": len(missing_days),
				"nan_days_count": len(nan_days),
				"filled_days_count": len(filled_days),
				"days_with_valid_data": recorded_days,
				"days_without_valid_data": total_days - recorded_days,
				"completeness_percentage": completeness_percentage
			})

		# Convertir resultados a DataFrame
		results_df= pd.DataFrame(results)

		# Clasificar estaciones por nivel de completitud
		bins = [0, 10, 50, 90, 99, 100]
		labels = ["<10%", "10-50%", "50-90%", "90-99%", "99-100%"]
		results_df['completeness_category'] = pd.cut(results_df['completeness_percentage'], bins=bins, labels=labels)

		# Guardar en un archivo CSV
		results_df.to_csv("stations_completeness.csv", index=False)
		print('stations_completeness.csv has been saved')
		
		# Graficar el mapa
		plt.figure(figsize=(10, 8))
		# Crear el eje con una proyección
		ax = plt.axes(projection=ccrs.PlateCarree())
		# Añadir líneas de costa y fronteras
		ax.coastlines(resolution='10m', linewidth=1)  # Líneas de costa
		ax.add_feature(cfeature.BORDERS, linestyle='--')
		# Graficar los datos
		scatter = ax.scatter(
			results_df['longitude'], results_df['latitude'],
			c=results_df['completeness_percentage'], cmap='viridis', s=100, edgecolor='k', vmin=0, vmax=100,
			transform=ccrs.PlateCarree()  # Transformación a coordenadas geográficas
		)
		# Añadir la barra de color
		colorbar = plt.colorbar(scatter, ax=ax, label="completeness percentage (%)")
		colorbar.set_ticks([0, 20, 40, 60, 80, 100])
		# Añadir título y etiquetas
		plt.title("Stations completeness percentage")
		plt.xlabel("Longitude")
		plt.ylabel("Latitude")
		# Guardar el archivo
		plt.savefig("stations_completeness_map.png")
		print('stations_completeness_map.png has been saved')
		# Cerrar la figura
		plt.close()
		
		# Contar estaciones por categoría
		summary = results_df['completeness_category'].value_counts().sort_index()
		total_stations = len(results_df)
		print("completeness summary:")
		for label, count in summary.items():
			percentage = (count / total_stations) * 100
			print(f"{label}: {count} stations ({percentage:.2f}%)")
		
		del stations_data_1
		del results
		del results_df
		
		# Filtrar filas eliminando las que tienen nan
		stations_data_0 = stations_data_0.dropna()
		# Filtrar filas eliminando las que tienen -99, -999 o -9999'
		values_to_remove = [-99, -999, -9999]
		stations_data_0 = stations_data_0[~stations_data_0[selected_variable].isin(values_to_remove)]
			
	except FileNotFoundError:
		print(f'Error - file not found: stations_data_{selected_variable}.csv')
		exit()
	
	for grid in selected_grids:
		stations_data = stations_data_0 # se crea un nuevo dataframe que contiene los datos de las estaciones del CSV 
		# Cargar datos de la rejilla (ISIMIP-CHELSA, CHIRTS, CHIRPS, ERA5, ERA5-land, 'COSMO-REA6', 'CERRA', 'CERRA-Land', 'EOBS') usando netCDF4
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
		
		if grid in ['ERA5-Land'] and selected_variable in ['precipitation']:
			targetvar = [string for string in names if string in ['tp']][0]
		if grid in ['EOBS_HR'] and selected_variable in ['precipitation']:
			targetvar = [string for string in names if string in ['rr']][0]
		if grid in ['EOBS_LR'] and selected_variable in ['precipitation']:
			targetvar = [string for string in names if string in ['rr']][0]
		if grid in ['ERA5'] and selected_variable in ['precipitation']:
			targetvar = [string for string in names if string in ['tp']][0]
		if grid in ['CERRA'] and selected_variable in ['precipitation']:
			targetvar = [string for string in names if string in ['var61']][0]	
		if grid == 'COSMO-REA6' and selected_variable == 'wind_speed':
    			targetvar = [string for string in names if string == 'var33'][0]
		if grid == 'COSMO-REA6' and selected_variable == 'humidity':
			targetvar = [string for string in names if string == 'var52'][0]
		elif grid in ['ERA5-Land'] and selected_variable == 'wind_speed':
    			targetvar = [string for string in names if string == 'u10'][0]
    			if not targetvar:
        			raise ValueError(f"No se encontraron variables de viento ('u10', 'v10') en: {names}")
		elif grid in ['ERA5-Land'] and selected_variable == 'humidity':
			targetvar = [string for string in names if string == 'hr2m'][0]
			if not targetvar:
				raise ValueError(f"No se encontraron variables de humedad en: {names}")
		elif grid in ['ERA5-Land'] and selected_variable not in ['precipitation', 'wind_speed', 'humidity']:
    			targetvar = [string for string in names if string in ['t2m']][0]
		else:
			targetvar = names[-1]
		targetlat = [string for string in names if 'lat' in string][0]
		targetlon = [string for string in names if 'lon' in string][0]
		targettime = [string for string in names if string in ['time', 'valid_time']][0]
		
		# Variables dentro del archivo netCDF y conversión de las unidades
		grid_lat = grid_data.variables[targetlat][:]
		grid_lon = grid_data.variables[targetlon][:]
		grid_time = grid_data.variables[targettime][:]
		
		try:
			if not np.any(grid_time.mask) == True:
				units = grid_data.variables[targettime].units
			else:
				units = 'days since 1991-01-01 00:00:00'
				grid_time = np.array(list(range(grid_time.shape[0])))
		except:
			if grid == 'CHIRTS':
				units = 'days since 1980-01-01 00:00:00'
			else:
				print('Error - units not found in netCDF metadata')
		
		if selected_variable in ['temperature', 'maximum_temperature', 'minimum_temperature'] and grid not in ['CHIRTS', 'ERA5']:
			grid_targetvar = grid_data.variables[targetvar][:].astype('float32') - 273.15 # convierte grados Kelvin a grados Celsius
		elif selected_variable in ['temperature', 'maximum_temperature', 'minimum_temperature'] and grid in ['CHIRTS', 'ERA5']:
			grid_targetvar = grid_data.variables[targetvar][:].astype('float32') # mantiene las unidades en grados Celsius
		elif selected_variable in ['wind_speed'] and grid not in ['COSMO-REA6']:
			grid_targetvar = grid_data.variables[targetvar][:].astype('float32') # mantiene las unidades de viento
		elif selected_variable in ['wind_speed'] and grid in ['COSMO-REA6']:
			grid_targetvar = grid_data.variables[targetvar][:][:,0,:,:].astype('float32') # mantiene las unidades de viento y pasa de 4D a 3D
		elif selected_variable == 'precipitation' and grid not in ['ISIMIP-CHELSA', 'CHIRPS', 'CERRA']:
			grid_targetvar = grid_data.variables[targetvar][:].astype('float32')*1000 # convierte m/día a mm/día
		elif selected_variable == 'precipitation' and grid == 'CERRA':
			grid_targetvar = grid_data.variables[targetvar][:][:,0,:,:].astype('float32')*1000 # convierte m/día a mm/día y pasa de 4D a 3D
		elif selected_variable == 'precipitation' and grid in ['CHIRPS']:
			grid_targetvar = grid_data.variables[targetvar][:].astype('float32') # mantiene las unidades en mm/día
		elif selected_variable == 'precipitation' and grid in ['ISIMIP-CHELSA']:
			grid_targetvar = grid_data.variables[targetvar][:].astype('float32')*86400 # convierte kg/m²s a mm/día
		elif selected_variable in ['humidity'] and grid not in ['COSMO-REA6']:
			grid_targetvar = grid_data.variables[targetvar][:].astype('float32')# mantiene las unidades de humedad
		elif selected_variable in ['humidity'] and grid in ['COSMO-REA6']:
			grid_targetvar = grid_data.variables[targetvar][:][:,0,:,:].astype('float32') # mantiene las unidades de humedad y pasa de 4D a 3D
		else:
			print('Error - units')
			exit()
			
		# forzar sentido ascendente
		if (grid_lat[0] > grid_lat[-1]):
			grid_lat = np.flip(grid_lat)
			grid_targetvar = np.flip(grid_targetvar, axis=1)
		if (grid_lon[0] > grid_lon[-1]):
			grid_lon = np.flip(grid_lon)
			grid_targetvar = np.flip(grid_targetvar, axis=2)
		# tratamiento de máscaras, nans y valores de relleno
		grid_targetvar = np.where(grid_targetvar.mask, np.nan, grid_targetvar) # valores enmascarados se sustituyen por NaN
		fill_value = getattr(grid_data.variables[targetvar], '_FillValue', None)  # Detectar si existe _FillValue
		grid_targetvar = np.where(grid_targetvar == fill_value, np.nan, grid_targetvar) # _FillValue se sustituye por NaN
		grid_data.close() # se cierra el netCDF
		del grid_data
		
		# Función para convertir fechas a índices de tiempo en la rejilla 
		def convert_time_to_index(time_array, date):
			# Convertir la fecha objetivo (date) a un número en las unidades originales
			time_num = nc.date2num(date, units=units, calendar='standard')
			# Calcular la diferencia en unidades (en el mismo sistema de unidades)
			time_diff = np.abs(time_array - time_num)
			# Verificar si alguna diferencia es mayor a 24 horas, en unidades compatibles
			if units.startswith('days'):
				threshold = 1  # 1 día
			elif units.startswith('hours'):
				threshold = 24  # 24 horas
			elif units.startswith('seconds'):
				threshold = 24 * 3600  # 24 horas * 3600 segundos
			else:
				raise ValueError("Unidades no soportadas en el cálculo.")
			# Si la diferencia es mayor que el umbral, devolver NaN
			if np.all(time_diff > threshold):
				return np.nan
			# Hacer la interpolación si las diferencias son menores o iguales a 24 horas
			time_idx = np.interp(time_num, time_array, np.arange(len(time_array)))
			return time_idx

		# Crear el interpolador para los datos de la rejilla 
		def create_interpolator(targetvar_data, lat_array, lon_array, lat_station, lon_station, time_idx, interpolation_method):
			lat_array = np.sort(np.unique(lat_array))
			lon_array = np.sort(np.unique(lon_array))		
			lat_idx = np.abs(lat_array - lat_station).argmin()
			lon_idx = np.abs(lon_array - lon_station).argmin()

			# Definir rangos para interpolación local
			lat_range = lat_array[max(0, lat_idx-1):min(len(lat_array), lat_idx+2)]
			lon_range = lon_array[max(0, lon_idx-1):min(len(lon_array), lon_idx+2)]	
			targetvar_range = targetvar_data[time_idx, max(0, lat_idx-1):min(len(lat_array), lat_idx+2), max(0, lon_idx-1):min(len(lon_array), lon_idx+2)]	
			
			# Comprobar y manejar `np.nan` en targetvar_range. Si todos son `np.nan`, devuelve `np.nan`
			if np.isnan(targetvar_range).all():
				targetvar_range[:] = np.nan  # Por ejemplo, rellena todo con ceros
				
			# Comprobar y manejar `np.nan` en targetvar_range. Si hay solamente algún `np.nan`, lo rellena con el valor válido más cercano al centro
			if np.isnan(targetvar_range).any() and not np.isnan(targetvar_range).all():
				center_idx = (targetvar_range.shape[0] // 2, targetvar_range.shape[1] // 2)  # Centro aproximado
				valid_mask = ~np.isnan(targetvar_range)  # Máscara de valores válidos (no-NaN)
				valid_indices = np.argwhere(valid_mask)  # Índices de valores válidos
				
				# Distancia al centro desde los valores válidos
				distances = np.linalg.norm(valid_indices - np.array(center_idx), axis=1)
				closest_valid_idx = valid_indices[distances.argmin()]  # Índice del valor más cercano válido
				
				# Rellenar NaN con el valor más cercano válido
				targetvar_range[~valid_mask] = targetvar_range[tuple(closest_valid_idx)]
			
			return RegularGridInterpolator(
				(lat_range, lon_range),
				targetvar_range,
				method=interpolation_method,
				bounds_error=True,
				fill_value=np.nan
			)
			
		# Función para extraer valores interpolados de la rejilla para las ubicaciones y fechas de las estaciones 
		def extract_interpolated_grid_value(lat, lon, date):
			time_idx = convert_time_to_index(grid_time, date)
			try: 
				interpolator = create_interpolator(grid_targetvar, grid_lat, grid_lon, lat, lon, int(time_idx),interpolation_method)
				interpolated_value = interpolator((lat, lon)) 
				return interpolated_value
			except:
				return np.nan	
			
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

		def calculate_metrics_interpolated_wspeed(data):
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
			percentile95_bias = np.percentile(data['interpolated_grid_value'], 95) - np.percentile(data[selected_variable], 95)
			
			return pd.Series({
				'Mean Bias': mean_bias,
				'P95 Bias': percentile95_bias,
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
		def calculate_metrics_interpolated_humidity(data):
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
			percentile95_bias = np.percentile(data['interpolated_grid_value'], 95) - np.percentile(data[selected_variable], 95)
			
			return pd.Series({
				'Mean Bias': mean_bias,
				'P95 Bias': percentile95_bias,
				'P90 Bias': percentile90_bias,
				'P10 Bias': percentile10_bias,
				'Mean Absolute Error': mean_absolute_error,
				'RMSE': rmse,
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

		elif selected_variable == 'wind_speed':
			metrics_per_station_interpolated = stations_data.groupby('station_id').apply(calculate_metrics_interpolated_wspeed).reset_index()
			# Guardar métricas para cada estación en un csv
			metrics_per_station_interpolated.to_csv('metrics_per_station_interpolated_' + grid + '_' + selected_variable + '.csv', index=False)
			print('metrics_per_station_interpolated_' + grid + '_' + selected_variable + '.csv has been saved')	
			units = {
				'Mean Bias': 'm/s',
				'P95 Bias': 'm/s',
				'P90 Bias': 'm/s',
				'P10 Bias': 'm/s',
				'Mean Absolute Error': 'm/s',
				'RMSE': 'm/s',
				'Correlation': 'Dimensionless',
				'Variance Bias': 'm²/s²'
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
			metrics_per_station_interpolated['Number of wet days Bias'] = metrics_per_station_accumulated['Number of wet days Bias'] / (stations_data['date'].nunique() / (365 * len(selected_months) / 12))

			metrics_per_station_interpolated.to_csv('metrics_per_station_interpolated_' + grid + '_' + selected_variable + '.csv', index=False)
			print('metrics_per_station_interpolated_' + grid + '_' + selected_variable + '.csv has been saved')

		elif selected_variable == 'humidity':
			metrics_per_station_interpolated = stations_data.groupby('station_id').apply(calculate_metrics_interpolated_humidity).reset_index()
                        # Guardar métricas para cada estación en un csv
			metrics_per_station_interpolated.to_csv('metrics_per_station_interpolated_' + grid + '_' + selected_variable + '.csv', index=False)
			print('metrics_per_station_interpolated_' + grid + '_' + selected_variable + '.csv has been saved')
			units = {
				'Mean Bias': '%',
				'P95 Bias': '%',
				'P90 Bias': '%',
				'P10 Bias': '%',
				'Mean Absolute Error': '%',
				'RMSE': '%',
				'Correlation': 'Dimensionless',
				'Variance Bias': '%²'
			}
			
		else:
			print(f'Error: variable {selected_variable} no contemplada para el cálculo de métricas.')
			continue

		# Calcular el ciclo anual del grid
		
		dfac = stations_data.dropna()
		dfac['month'] = dfac['date'].dt.month
		dfac['interpolated_grid_value'] = dfac['interpolated_grid_value'].astype(float)
		# Agrupar por mes y calcular el ciclo anual promediando con todas las estaciones observacionales
		if selected_variable != 'precipitation':
			# Paso 1: Calcular la media mensual por estación
			station_monthly = dfac.groupby(['station_id', 'month']).agg({
				selected_variable: 'mean',
				'interpolated_grid_value': 'mean'
			}).reset_index()
			  
			# Paso 2: Calcular la media mensual global entre estaciones
			monthly_avg = station_monthly.groupby('month').agg({
				selected_variable: 'mean',
				'interpolated_grid_value': 'mean'
			}).reset_index()
		else:
			# Paso 1: Calcular la suma mensual por estación
			station_monthly = dfac.groupby(['station_id', 'month']).agg({
				selected_variable: 'sum',
				'interpolated_grid_value': 'sum'
			}).reset_index()
			
			# Paso 2: Dividir por el número de años en el periodo de análisis
			num_years = stations_data['date'].nunique() / (365 * len(selected_months) / 12)  
			station_monthly[selected_variable] /= num_years
			station_monthly['interpolated_grid_value'] /= num_years
			
			# Paso 3: Calcular la media mensual global entre todas las estaciones
			monthly_avg = station_monthly.groupby('month').agg({
				selected_variable: 'mean',
				'interpolated_grid_value': 'mean'
			}).reset_index()
		# Exportar a CSV
		monthly_avg.to_csv(selected_variable + '_' + grid + '_annual_cycle_comparison.csv', index=False)
		
		del stations_data
		del stations_data_accu_ini
		del dfac
	
	# Calcular el ciclo anual de las observaciones
	
	stations_data_0['month'] = stations_data_0['date'].dt.month
	stations_data_0[selected_variable] = pd.to_numeric(stations_data_0[selected_variable], errors='coerce')
	stations_data_0[selected_variable] = stations_data_0[selected_variable].astype(float)
	# Agrupar por mes y calcular el ciclo anual promediando con todas las estaciones observacionales
	if selected_variable != 'precipitation':
		# Paso 1: Calcular la media mensual por estación
		station_monthly_obs = stations_data_0.groupby(['station_id', 'month']).agg({
			selected_variable: 'mean',
			'interpolated_grid_value': 'mean'
		}).reset_index()
			
		# Paso 2: Calcular la media mensual global entre estaciones
		monthly_avg_obs = station_monthly_obs.groupby('month').agg({
			selected_variable: 'mean',
			'interpolated_grid_value': 'mean'
		}).reset_index()
	else:
		# Paso 1: Calcular la suma mensual por estación
		station_monthly_obs = stations_data_0.groupby(['station_id', 'month']).agg({
			selected_variable: 'sum',
			'interpolated_grid_value': 'sum'
		}).reset_index()
		
		# Paso 2: Dividir por el número de años en el periodo de análisis
		num_years = stations_data_0['date'].nunique() / (365 * len(selected_months) / 12)
		station_monthly_obs[selected_variable] /= num_years
		
		# Paso 3: Calcular la media mensual global entre todas las estaciones
		monthly_avg_obs = station_monthly_obs.groupby('month').agg({
			selected_variable: 'mean',
			'interpolated_grid_value': 'mean'
		}).reset_index()
	# Exportar a CSV
	monthly_avg_obs.to_csv(selected_variable + '_' + grid + '_annual_cycle_obs.csv', index=False)

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
			
			# Generar mapas para cada métrica
			# Cargar los datos de los CSVs
			stations_df = pd.read_csv("stations_completeness.csv")
			# Unir los dos DataFrames por station_id
			merged_df = pd.merge(stations_df, metrics_data, on="station_id", how="inner")
			metrics_to_plot = [col for col in metrics_data.columns if col != "station_id"]
			# Generar un scatterplot para cada métrica
			for metric in metrics_to_plot:
				# Configurar proyección y crear figura
				fig, ax = plt.subplots(
					figsize=(10, 8),
					subplot_kw={'projection': ccrs.PlateCarree()}
				)
				# Agregar la línea de costa con alta resolución
				ax.coastlines(resolution='10m', color='black', linewidth=1)
				# Agregar fronteras
				ax.add_feature(cfeature.BORDERS, linestyle='--')
				# Graficar los datos de las estaciones
				scatter = ax.scatter(
					merged_df['longitude'], merged_df['latitude'],
					c=merged_df[metric], cmap='viridis', s=100, edgecolor='k', transform=ccrs.PlateCarree()
				)
				# Configurar la barra de color
				colorbar = plt.colorbar(scatter, ax=ax, label=f'{metric} ({units.get(metric, "")})')
				# Títulos y etiquetas
				plt.title(f"{metric} comparison between {grid} and stations ")
				plt.xlabel("Longitude")
				plt.ylabel("Latitude")
				# Guardar y mostrar el gráfico
				plt.savefig(f"map_{metric.replace(' ', '_').lower()}_{grid}_{selected_variable}.png")
				#plt.show()
				plt.close()
			
		except FileNotFoundError:
			print(f'Warning: No metrics file found for {grid} and {selected_variable}')
	
	# Cargar el ciclo anual de las observaciones desde su CSV
	file_name_annual_cycle_obs = selected_variable + '_' + grid + '_annual_cycle_obs.csv'
	annual_cycle_data_obs = pd.read_csv(file_name_annual_cycle_obs)

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
	plt.plot(annual_cycle_dict[grid]['month'], annual_cycle_data_obs[selected_variable], label= 'observations', marker='o', color='black')#
	for grid in selected_grids:
		plt.plot(annual_cycle_dict[grid]['month'], annual_cycle_dict[grid]['interpolated_grid_value'], label= grid, marker='o')
	# Configurar el gráfico
	plt.xlabel('Month')
	if selected_variable == 'precipitation':
		plt.ylabel(selected_variable + ' (mm)')
	elif selected_variable == 'wind_speed':
		plt.ylabel(selected_variable + ' (m/s)')
	else:
		plt.ylabel(selected_variable + ' (°C)')
	plt.title('Average Annual Cycle')
	# Etiquetas de los meses abreviados en inglés
	months_abbr = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
	months_names = [months_abbr[month - 1] for month in selected_months]
	# Usar plt.xticks() para asignar etiquetas de texto a los meses
	plt.xticks(annual_cycle_dict[grid]['month'], months_names)
	plt.legend()
	plt.savefig(f'{selected_variable}_annual_cycle_grids_comparison.png')
	plt.close()
	print(f'{selected_variable}_annual_cycle_grids_comparison.png has been saved')
	
	
def on_generate_button_click():
	selected_variables = [combo_variable.get()]
	selected_grids = listbox_grids.curselection()
	selected_grids = [grids[i] for i in selected_grids]
	interpolation_method = interpolation_var.get()
	start_year = entry_start_year.get()
	end_year = entry_end_year.get()
	period_type = period_var.get()

	# Validar entradas de año
	if not start_year.isdigit() or not end_year.isdigit():
		messagebox.showerror("Error", "Por favor ingresa años válidos.")
		return
	if int(start_year) > int(end_year):
		messagebox.showerror("Error", "El año de inicio debe ser menor o igual al año de fin.")
		return
	if period_type == "Anual":
		selected_months = list(range(1, 13))  # Todos los meses
	else:
		selected_months = [i + 1 for i, var in enumerate(month_vars) if var.get() == 1]  # Meses seleccionados
		
	generate_metrics_and_plots(selected_grids, selected_variables[0], int(start_year), int(end_year), selected_months, interpolation_method)
	
# Variables y lista de rejillas
variables = ['temperature', 'maximum_temperature', 'minimum_temperature', 'precipitation', 'wind_speed', 'humidity']
grids = ['ISIMIP-CHELSA', 'CHIRTS', 'CHIRPS', 'ERA5', 'ERA5-Land', 'COSMO-REA6', 'CERRA', 'CERRA-Land', 'EOBS', 'EOBS_HR', 'EOBS_LR']
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']


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

# Selección del método de interpolación
label_interpolation = ttk.Label(root, text='Select Interpolation Method:')
label_interpolation.pack(pady=10)

interpolation_var = tk.StringVar(value="nearest")  # Valor por defecto

frame_interpolation = ttk.Frame(root)
frame_interpolation.pack(pady=10)

radiobutton_nearest = ttk.Radiobutton(frame_interpolation, text="Nearest Neighbor", variable=interpolation_var, value="nearest")
radiobutton_nearest.grid(row=0, column=0, padx=5)

radiobutton_bilinear = ttk.Radiobutton(frame_interpolation, text="Bilinear", variable=interpolation_var, value="linear")
radiobutton_bilinear.grid(row=0, column=1, padx=5)

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

# Selección de tipo de periodo
period_var = tk.StringVar(value="Anual")

frame_seasonal = ttk.Frame(root)
frame_seasonal.pack(pady=10)

radiobutton_annual = ttk.Radiobutton(frame_seasonal, text="Annual", variable=period_var, value="Anual")
radiobutton_annual.grid(row=0, column=0, padx=5)

radiobutton_monthly = ttk.Radiobutton(frame_seasonal, text="Custom Months", variable=period_var, value="Mensual")
radiobutton_monthly.grid(row=0, column=1, padx=5)

# Checkboxes para seleccionar meses
frame_months = ttk.Frame(root)
frame_months.pack(pady=10)

month_vars = []
for i, month in enumerate(months):
    var = tk.IntVar(value=0)
    month_vars.append(var)
    checkbutton = ttk.Checkbutton(frame_months, text=month, variable=var)
    checkbutton.grid(row=i // 4, column=i % 4, sticky="w", padx=5)    

# Botón para generar las métricas y gráficos
generate_button = ttk.Button(root, text='Generate Metrics & Plots', command=on_generate_button_click)
generate_button.pack(pady=20)

# Iniciar la interfaz gráfica
root.mainloop()
