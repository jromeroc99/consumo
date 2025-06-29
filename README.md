# Proyecto Consumo Energético

## Estructura del Proyecto

- `datos/`: Scripts y utilidades para descarga, limpieza y procesamiento de datos.
- `entrenamiento/`: Modelos de machine learning, entrenamiento y evaluación.
- `combina_mongo.py`: Script original de procesamiento de datos (será migrado a `datos/`).
- `datos_limpios.csv`: Ejemplo de datos procesados.
- `requirements.txt`: Dependencias del proyecto.

## Flujo de trabajo

1. Procesa y limpia los datos con los scripts en `datos/`.
2. Entrena modelos en `entrenamiento/` usando los datos limpios.
3. Evalúa y guarda los modelos entrenados.

## Objetivo

Predecir las columnas `fase0`, `fase1` y `fase2` a partir del resto de variables del dataset limpio usando modelos de machine learning (ejemplo: PyTorch).