# grid_evaluator

Este programa está diseñado para calcular métricas de diferencias y generar gráficos de comparación entre datos climáticos de rejillas (grids) y datos de estaciones observacionales. La interfaz gráfica de usuario (GUI) permite seleccionar entre diferentes rejillas y variables climáticas, y visualizar las métricas y gráficos generados.

Requisitos del Sistema

    Python 3.x
    Bibliotecas: tkinter, netCDF4, pandas, numpy, seaborn, matplotlib, scipy

Instrucciones de Uso

1. Asegúrese de tener instaladas las bibliotecas necesarias: 

	pip install netCDF4 pandas numpy seaborn matplotlib scipy

2. Coloque los archivos necesarios en el mismo directorio que el script del programa.

3. Ejecute el script en su entorno de Python:

	python <nombre_del_script>.py

4. Use la interfaz gráfica para seleccionar las rejillas y variables que desea analizar.

5. Haga clic en el botón para generar métricas y gráficos. Los resultados se guardarán en archivos CSV y PNG en el directorio actual.


Archivos Requeridos

Archivos netCDF para las Rejillas

Los archivos netCDF deben estar nombrados y formateados de la siguiente manera:

Formato de nombre de archivo: grid_data_<GRID>_<VARIABLE>.nc
<GRID>: Nombre de la rejilla (por ejemplo, ISIMIP-CHELSA, CHIRTS, CHIRPS, ERA5, ERA5-Land)
<VARIABLE>: Nombre de la variable climática (por ejemplo, temperature, maximum_temperature, minimum_temperature, precipitation)

Ejemplos:

    grid_data_ISIMIP-CHELSA_temperature.nc
    grid_data_CHIRTS_maximum_temperature.nc
    grid_data_CHIRPS_precipitation.nc
    grid_data_ERA5_temperature.nc
    grid_data_ERA5-Land_minimum_temperature.nc

Archivos CSV para los Datos de las Estaciones

Los archivos CSV deben estar nombrados y formateados de la siguiente manera:

Formato de nombre de archivo: stations_data_<VARIABLE>.csv
	<VARIABLE>: Nombre de la variable climática (por ejemplo, temperature, maximum_temperature, minimum_temperature, precipitation)

Ejemplos:

    stations_data_temperature.csv
    stations_data_maximum_temperature.csv
    stations_data_minimum_temperature.csv
    stations_data_precipitation.csv

Formato del archivo CSV:

El archivo CSV debe contener las siguientes columnas:

    station_id: Identificador de la estación
    latitude: Latitud de la estación
    longitude: Longitud de la estación
    date: Fecha de la observación (en formato YYYY-MM-DD comenzando en 1991-01-01)
    <VARIABLE>: Valor observado de la variable climática (por ejemplo, temperature, maximum_temperature, minimum_temperature, precipitation)

Las filas deben ordenarse primero por estación (station_id) y después por fecha (date).

Ejemplo de contenido de stations_data_temperature.csv:

	station_id,latitude,longitude,date,temperature
	1,35.6895,139.6917,1991-01-01,25.0
	1,35.6895,139.6917,1991-01-02,25.7
	1,35.6895,139.6917,1991-01-03,29.0
	1,35.6895,139.6917,1991-01-04,22.6
	1,35.6895,139.6917,1991-01-05,23.1
	1,35.6895,139.6917,1991-01-06,25.4
	...
	2,34.0522,118.2437,1991-01-01,10.0
	2,34.0522,118.2437,1991-01-02,15.0
	2,34.0522,118.2437,1991-01-03,17.9
	2,34.0522,118.2437,1991-01-04,10.0
	2,34.0522,118.2437,1991-01-05,24.5
	2,34.0522,118.2437,1991-01-06,21.3
	...
	3,51.5074,-0.1278,1991-01-01,27.0
	3,51.5074,-0.1278,1991-01-02,29.1
	3,51.5074,-0.1278,1991-01-03,18.5
	...


Descripción del Programa

El programa carga los datos de las rejillas desde archivos netCDF y los datos de las estaciones desde archivos CSV. Luego, calcula métricas de diferencia entre los valores interpolados de las rejillas y los valores observados en las estaciones. Las métricas calculadas incluyen sesgo medio, error absoluto medio, RMSE, correlación, sesgo de varianza, y otros.

Las métricas calculadas se guardan en archivos CSV y los gráficos de comparación se guardan en archivos PNG.
Contacto

Advertencia: el programa dará error si se intenta comparar rejillas para una variable que no contienen. Por ejemplo, si el usuario/a selecciona las rejillas 'ISIMIP-CHELSA' y 'CHIRTS', y selecciona como variable 'precipitation', el programa dará error en tanto que 'CHIRTS' no contiene información de la variable 'precipitation'.

Para más información o consultas, por favor contacte a ccorreag@aemet.es

