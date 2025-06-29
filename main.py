from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
import torch
import pvlib

LATITUD = 36.7347194
LONGITUD = -4.3563086
FESTIVOS_ES = set([
    pd.to_datetime(f).date() for f in [
        '2023-01-01', '2023-01-06', '2023-04-07', '2023-05-01', '2023-08-15',
        '2023-10-12', '2023-11-01', '2023-12-06', '2023-12-08', '2023-12-25',
        '2024-01-01', '2024-01-06', '2024-03-29', '2024-05-01', '2024-08-15',
        '2024-10-12', '2024-11-01', '2024-12-06', '2024-12-08', '2024-12-25',
        '2025-01-01', '2025-01-06', '2025-04-18', '2025-05-01', '2025-08-15',
        '2025-10-12', '2025-11-01', '2025-12-06', '2025-12-08', '2025-12-25',
    ]
])

# --- Modelo PyTorch ---
from entrenamiento.modelo_pytorch import MLP

scaler_X = joblib.load('scaler_X.pkl')
scaler_y = joblib.load('scaler_y.pkl')
input_dim = scaler_X.mean_.shape[0]
output_dim = 3
model = MLP(input_dim=input_dim, output_dim=output_dim)
model.load_state_dict(torch.load('modelo_consumo.pth', map_location=torch.device('cpu')))
model.eval()

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # O restringe a ["http://localhost:5173"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    fecha: str  # "YYYY-MM-DD"
    hora: str   # "HH:MM"
    temperatura: float
    humedad: float
    Papa: bool
    Mama: bool
    Ipe: bool
    Javi: bool

@app.post("/predict")
def predict(data: PredictRequest):
    dt = datetime.strptime(f"{data.fecha} {data.hora}", "%Y-%m-%d %H:%M")
    unix = int(dt.timestamp())
    dia = dt.day
    mes = dt.month
    hora_ = dt.hour
    minuto = dt.minute
    weekday = dt.weekday()
    dayofyear = dt.timetuple().tm_yday
    hora_sin = np.sin(2 * np.pi * hora_ / 24)
    hora_cos = np.cos(2 * np.pi * hora_ / 24)
    semana_sin = np.sin(2 * np.pi * weekday / 7)
    semana_cos = np.cos(2 * np.pi * weekday / 7)
    mes_sin = np.sin(2 * np.pi * mes / 12)
    mes_cos = np.cos(2 * np.pi * mes / 12)
    anio_sin = np.sin(2 * np.pi * dayofyear / 366)
    anio_cos = np.cos(2 * np.pi * dayofyear / 366)
    festivo = 1 if dt.date() in FESTIVOS_ES else 0
    solpos = pvlib.solarposition.get_solarposition(pd.DatetimeIndex([dt]), LATITUD, LONGITUD)
    alturaSol = float(np.asarray(solpos['elevation'])[0])
    columnas = [
        'unix', 'dia', 'mes', 'hora', 'minuto',
        'hora_sin', 'hora_cos', 'semana_sin', 'semana_cos',
        'mes_sin', 'mes_cos', 'anio_sin', 'anio_cos',
        'festivo', 'alturaSol', 'temperatura', 'humedad',
        'Papa', 'Mama', 'Ipe', 'Javi'
    ]
    valores = [
        unix, dia, mes, hora_, minuto,
        hora_sin, hora_cos, semana_sin, semana_cos,
        mes_sin, mes_cos, anio_sin, anio_cos,
        festivo, alturaSol, data.temperatura, data.humedad,
        int(data.Papa), int(data.Mama), int(data.Ipe), int(data.Javi)
    ]
    X_input = pd.DataFrame([valores], columns=pd.Index(columnas))
    X_scaled = scaler_X.transform(X_input)
    with torch.no_grad():
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_pred_scaled = model(X_tensor).numpy()
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
    return {
        "fase0": float(y_pred[0][0]),
        "fase1": float(y_pred[0][1]),
        "fase2": float(y_pred[0][2])
    } 