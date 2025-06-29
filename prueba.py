import torch
import pandas as pd
import numpy as np
import joblib
import pvlib
from datetime import datetime
from entrenamiento.modelo_pytorch import MLP
import tkinter as tk
from tkinter import ttk, messagebox

# Constantes de localización
LATITUD = 36.7347194
LONGITUD = -4.3563086

# Festivos nacionales (idéntico a procesamiento.py)
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

# Cargar scalers y modelo
scaler_X = joblib.load('scaler_X.pkl')
scaler_y = joblib.load('scaler_y.pkl')

# Input: datetime, temperatura, humedad, personas
input_datetime = datetime(2023, 4, 1, 12, 0)  # <-- Modifica aquí la fecha/hora
input_temp = 22.0  # Temperatura en ºC
input_hum = 55.0   # Humedad en %
personas = 11      # Número de personas

# Variables temporales
unix = int(input_datetime.timestamp())
dia = input_datetime.day
mes = input_datetime.month
hora = input_datetime.hour
minuto = input_datetime.minute
weekday = input_datetime.weekday()
dayofyear = input_datetime.timetuple().tm_yday

# Variables cíclicas
hora_sin = np.sin(2 * np.pi * hora / 24)
hora_cos = np.cos(2 * np.pi * hora / 24)
semana_sin = np.sin(2 * np.pi * weekday / 7)
semana_cos = np.cos(2 * np.pi * weekday / 7)
mes_sin = np.sin(2 * np.pi * mes / 12)
mes_cos = np.cos(2 * np.pi * mes / 12)
anio_sin = np.sin(2 * np.pi * dayofyear / 366)
anio_cos = np.cos(2 * np.pi * dayofyear / 366)

# Festivo
date_only = input_datetime.date()
festivo = 1 if date_only in FESTIVOS_ES else 0

# Altura del sol
solpos = pvlib.solarposition.get_solarposition(pd.DatetimeIndex([input_datetime]), LATITUD, LONGITUD)
alturaSol = float(np.asarray(solpos['elevation'])[0])

# Variables de personas (igual que procesamiento.py)
Papa = (personas // 8) % 2
Mama = (personas // 4) % 2
Ipe = (personas // 2) % 2
Javi = personas % 2

# Columnas numéricas usadas en el entrenamiento
columnas = [
    'unix', 'dia', 'mes', 'hora', 'minuto',
    'hora_sin', 'hora_cos', 'semana_sin', 'semana_cos',
    'mes_sin', 'mes_cos', 'anio_sin', 'anio_cos',
    'festivo', 'alturaSol', 'temperatura', 'humedad',
    'Papa', 'Mama', 'Ipe', 'Javi'
]

valores = [
    unix, dia, mes, hora, minuto,
    hora_sin, hora_cos, semana_sin, semana_cos,
    mes_sin, mes_cos, anio_sin, anio_cos,
    festivo, alturaSol, input_temp, input_hum,
    Papa, Mama, Ipe, Javi
]

assert len(columnas) == len(valores), 'columnas y valores deben tener la misma longitud'
X_input = pd.DataFrame([valores], columns=pd.Index(columnas))

# Escalar entrada
X_scaled = scaler_X.transform(X_input)

# Cargar modelo
input_dim = X_scaled.shape[1]
output_dim = 3  # fase0, fase1, fase2
model = MLP(input_dim=input_dim, output_dim=output_dim)
model.load_state_dict(torch.load('modelo_consumo.pth', map_location=torch.device('cpu')))
model.eval()

# Convertir a tensor y predecir
with torch.no_grad():
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_pred_scaled = model(X_tensor).numpy()
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

print('Predicción (fase0, fase1, fase2):', y_pred[0])

def calcular_prediccion():
    try:
        fecha = entry_fecha.get()
        hora = entry_hora.get()
        temp = float(entry_temp.get())
        hum = float(entry_hum.get())
        Papa = int(var_papa.get())
        Mama = int(var_mama.get())
        Ipe = int(var_ipe.get())
        Javi = int(var_javi.get())
        dt = datetime.strptime(f"{fecha} {hora}", "%Y-%m-%d %H:%M")
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
            festivo, alturaSol, temp, hum,
            Papa, Mama, Ipe, Javi
        ]
        X_input = pd.DataFrame([valores], columns=pd.Index(columnas))
        X_scaled = scaler_X.transform(X_input)
        with torch.no_grad():
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
            y_pred_scaled = model(X_tensor).numpy()
            y_pred = scaler_y.inverse_transform(y_pred_scaled)
        resultado = f"Fase 0: {y_pred[0][0]:.2f}\nFase 1: {y_pred[0][1]:.2f}\nFase 2: {y_pred[0][2]:.2f}"
        label_resultado.config(text=resultado)
    except Exception as e:
        messagebox.showerror("Error", f"Error en la predicción: {e}")

root = tk.Tk()
root.title("Predicción Consumo Energético")
root.geometry("350x500")
root.resizable(False, False)

# Centrar ventana
def center(win):
    win.update_idletasks()
    width = win.winfo_width()
    height = win.winfo_height()
    x = (win.winfo_screenwidth() // 2) - (width // 2)
    y = (win.winfo_screenheight() // 2) - (height // 2)
    win.geometry(f"{width}x{height}+{x}+{y}")
center(root)

# Layout minimalista
title = ttk.Label(root, text="Predicción Consumo Energético", font=("Segoe UI", 15, "bold"))
title.pack(pady=(28, 18))

form = ttk.Frame(root)
form.pack(pady=0, padx=30, fill="x")

# Fecha
lbl_fecha = ttk.Label(form, text="Fecha (YYYY-MM-DD):", font=("Segoe UI", 10))
lbl_fecha.grid(row=0, column=0, sticky="w", pady=7)
entry_fecha = ttk.Entry(form, width=18)
entry_fecha.insert(0, datetime.now().strftime("%Y-%m-%d"))
entry_fecha.grid(row=0, column=1, pady=7)
# Hora
lbl_hora = ttk.Label(form, text="Hora (HH:MM):", font=("Segoe UI", 10))
lbl_hora.grid(row=1, column=0, sticky="w", pady=7)
entry_hora = ttk.Entry(form, width=18)
entry_hora.insert(0, datetime.now().strftime("%H:00"))
entry_hora.grid(row=1, column=1, pady=7)
# Temperatura
lbl_temp = ttk.Label(form, text="Temperatura (ºC):", font=("Segoe UI", 10))
lbl_temp.grid(row=2, column=0, sticky="w", pady=7)
entry_temp = ttk.Entry(form, width=18)
entry_temp.insert(0, "22.0")
entry_temp.grid(row=2, column=1, pady=7)
# Humedad
lbl_hum = ttk.Label(form, text="Humedad (%):", font=("Segoe UI", 10))
lbl_hum.grid(row=3, column=0, sticky="w", pady=7)
entry_hum = ttk.Entry(form, width=18)
entry_hum.insert(0, "55.0")
entry_hum.grid(row=3, column=1, pady=7)

# Toggles para Papa, Mama, Ipe, Javi
lbl_personas = ttk.Label(form, text="Personas presentes:", font=("Segoe UI", 10, "bold"))
lbl_personas.grid(row=4, column=0, sticky="w", pady=(14, 0))

var_papa = tk.BooleanVar(value=True)
var_mama = tk.BooleanVar(value=True)
var_ipe = tk.BooleanVar(value=False)
var_javi = tk.BooleanVar(value=True)

check_papa = ttk.Checkbutton(form, text="Papa", variable=var_papa)
check_papa.grid(row=5, column=0, sticky="w", pady=2)
check_mama = ttk.Checkbutton(form, text="Mama", variable=var_mama)
check_mama.grid(row=5, column=1, sticky="w", pady=2)
check_ipe = ttk.Checkbutton(form, text="Ipe", variable=var_ipe)
check_ipe.grid(row=6, column=0, sticky="w", pady=2)
check_javi = ttk.Checkbutton(form, text="Javi", variable=var_javi)
check_javi.grid(row=6, column=1, sticky="w", pady=2)

# Botón
btn = ttk.Button(root, text="Predecir Consumo", command=calcular_prediccion)
btn.pack(pady=(18, 10), ipadx=8, ipady=4)

# Resultado
label_resultado = ttk.Label(root, text="", font=("Segoe UI", 12), anchor="center", justify="center")
label_resultado.pack(pady=(18, 0), fill="x")

root.mainloop() 