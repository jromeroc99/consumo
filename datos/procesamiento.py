import pandas as pd
import numpy as np
import pvlib
from pymongo import MongoClient
from dotenv import load_dotenv
import os

# Carga variables de entorno desde .env
load_dotenv()

MONGO_URI = os.getenv('MONGO_URI')
DB_NAME = 'tfmJavi'
LATITUD, LONGITUD = 36.7347194, -4.3563086

# Lista de días festivos nacionales en España (formato YYYY-MM-DD)
FESTIVOS_ES = [
    '2023-01-01', '2023-01-06', '2023-04-07', '2023-05-01', '2023-08-15',
    '2023-10-12', '2023-11-01', '2023-12-06', '2023-12-08', '2023-12-25',
    '2024-01-01', '2024-01-06', '2024-03-29', '2024-05-01', '2024-08-15',
    '2024-10-12', '2024-11-01', '2024-12-06', '2024-12-08', '2024-12-25',
    '2025-01-01', '2025-01-06', '2025-04-18', '2025-05-01', '2025-08-15',
    '2025-10-12', '2025-11-01', '2025-12-06', '2025-12-08', '2025-12-25',
]
FESTIVOS_ES = set([pd.to_datetime(f).date() for f in FESTIVOS_ES])

class ProcesadorDatos:
    """
    Clase para descargar, limpiar, transformar y combinar datos de MongoDB para análisis energético.
    """
    def __init__(self, uri: str, db_name: str, lat: float, lon: float):
        self.uri = uri
        self.db_name = db_name
        self.lat = lat
        self.lon = lon

    def obtener_datos_mongo(self, col_name: str) -> pd.DataFrame:
        cliente = MongoClient(self.uri)
        db = cliente[self.db_name]
        datos = list(db[col_name].find())
        cliente.close()
        return pd.DataFrame(datos)

    def limpiar_ambientes(self, df_amb: pd.DataFrame) -> pd.DataFrame:
        df_amb = df_amb.copy()
        df_amb['_id'] = df_amb['_id'].astype(str)
        df_amb['festivo'] = df_amb['festivo'].astype(int)
        df_amb['personas'] = df_amb['personas'].astype('Int64')
        df_amb['cielo'] = df_amb['cielo'].astype(str).str.replace(r'[^0-9]', '', regex=True)
        df_amb['cielo'] = df_amb['cielo'].apply(lambda x: int(x) if x.strip() != '' else 0)
        df_amb['_id'] = df_amb['_id'].astype(str)
        return df_amb

    def limpiar_periodico(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if 'ambiente' in df.columns:
            df['ambiente'] = df['ambiente'].astype(str)
        if 'tiempo' in df.columns:
            df['tiempo'] = pd.to_datetime(df['tiempo'])
        return df

    def agregar_features_tiempo(self, df: pd.DataFrame) -> pd.DataFrame:
        df['unix'] = df['tiempo'].apply(lambda fecha: int(fecha.timestamp()))
        df['dia'] = df['tiempo'].dt.day
        df['mes'] = df['tiempo'].dt.month
        df['hora'] = df['tiempo'].dt.hour
        df['minuto'] = df['tiempo'].dt.minute
        return df

    def agregar_features_ciclicas(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['hora_sin'] = np.sin(2 * np.pi * df['hora'] / 24)
        df['hora_cos'] = np.cos(2 * np.pi * df['hora'] / 24)
        df['semana_sin'] = np.sin(2 * np.pi * df['tiempo'].dt.weekday / 7)
        df['semana_cos'] = np.cos(2 * np.pi * df['tiempo'].dt.weekday / 7)
        df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12)
        df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12)
        df['anio_sin'] = np.sin(2 * np.pi * df['tiempo'].dt.dayofyear / 366)
        df['anio_cos'] = np.cos(2 * np.pi * df['tiempo'].dt.dayofyear / 366)
        return df

    def combinar_ambientes(self, df: pd.DataFrame, df_amb: pd.DataFrame) -> pd.DataFrame:
        df = pd.merge(df, df_amb, left_on='ambiente', right_on='_id', how='left')
        for col in ["_id_x", "_id_y", "ambiente", "fechaUTC", 'cielo']:
            if col in df.columns:
                df = df.drop(col, axis=1)
        return df

    def calcular_posicion_solar(self, df: pd.DataFrame) -> pd.DataFrame:
        resultado = pvlib.solarposition.get_solarposition(df['tiempo'], self.lat, self.lon)
        if isinstance(resultado, pd.DataFrame):
            return resultado
        else:
            return pd.DataFrame(resultado)

    def limpiar_e_imputar(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in ['fase0', 'fase1', 'fase2']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        mask0 = (df['fase0'] == 0) & (df['fase1'] == 0) & (df['fase2'] == 0)
        mask365 = (df['fase0'] == 365) & (df['fase1'] == 72) & (df['fase2'] == 223)
        df.loc[mask0, ['fase0', 'fase1', 'fase2']] = np.nan
        df.loc[mask365, ['fase0', 'fase1', 'fase2']] = np.nan
        if 'personas' in df.columns:
            df['personas'] = df['personas'].fillna(15)
            df['Papa'] = (df['personas'] // 8) % 2
            df['Mama'] = (df['personas'] // 4) % 2
            df['Ipe'] = (df['personas'] // 2) % 2
            df['Javi'] = df['personas'] % 2
            df = df.drop(["personas"], axis=1)
        if 'festivo' in df.columns:
            df['festivo'] = df['festivo'].fillna(0)
        return df

    def rellenar_temp_humedad_por_mes(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if 'mes' not in df.columns:
            df['mes'] = df['tiempo'].dt.month
        temp_median = df.groupby('mes')['temperatura'].transform('median')
        hum_median = df.groupby('mes')['humedad'].transform('median')
        df['temperatura'] = df['temperatura'].fillna(temp_median)
        df['humedad'] = df['humedad'].fillna(hum_median)
        df['temperatura'] = df['temperatura'].fillna(df['temperatura'].median())
        df['humedad'] = df['humedad'].fillna(df['humedad'].median())
        return df

    def rellenar_festivos(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        mask_nan = df['festivo'].isna()
        fechas = df.loc[mask_nan, 'tiempo'].dt.date
        df.loc[mask_nan, 'festivo'] = fechas.apply(lambda d: 1.0 if d in FESTIVOS_ES else 0.0).astype(float)
        df['festivo'] = df['festivo'].astype(int)
        return df

    def obtener_df_limpia(self) -> pd.DataFrame:
        if not self.uri:
            raise ValueError('MONGO_URI no está definida en el entorno. Por favor, añade la variable al archivo .env')
        df = self.obtener_datos_mongo('periodic')
        df = df.drop(['fase0_r','fase1_r', 'fase2_r', 'fase0_e', 'fase1_e', 'fase2_e'], axis=1)
        df_amb = self.obtener_datos_mongo('ambientes')
        df_amb = self.limpiar_ambientes(df_amb)
        df = self.limpiar_periodico(df)
        df = self.agregar_features_tiempo(df)
        df = self.agregar_features_ciclicas(df)
        df = self.combinar_ambientes(df, df_amb)
        df = self.rellenar_temp_humedad_por_mes(df)
        solar_df = self.calcular_posicion_solar(df)
        df['alturaSol'] = solar_df['elevation'].values
        df = self.limpiar_e_imputar(df)
        df = self.rellenar_festivos(df)
        df_limpio = df.dropna()
        return df_limpio

if __name__ == "__main__":
    # Ejemplo de uso del procesador de datos
    try:
        if MONGO_URI is None:
            raise ValueError("MONGO_URI no está definida en el entorno. Por favor, añade la variable al archivo .env")
        # Instanciar el procesador con los parámetros de conexión y localización
        procesador = ProcesadorDatos(MONGO_URI, DB_NAME, LATITUD, LONGITUD)
        # Obtener el DataFrame limpio
        df_limpio = procesador.obtener_df_limpia()
        # Mostrar las primeras filas y el número total de filas limpias
        print(df_limpio.head())
        print(f"Filas totales limpias: {len(df_limpio)}")
        print("Columnas y tipos:")
        print(df_limpio.dtypes)
        
    except Exception as e:
        print(f"Error al procesar datos: {e}") 