# 🧹 Procesamiento de Datos - Consumo Energético

Esta carpeta contiene los scripts para descargar, limpiar, transformar y preparar los datos que alimentan los modelos de predicción. Aquí comienza el flujo de ciencia de datos del proyecto.

---

## 📂 Estructura

- `procesamiento.py` — Script principal para procesar y limpiar los datos
- `__init__.py` — Inicialización del módulo

---

## 🚦 ¿Qué hace `procesamiento.py`? (Paso a paso)

### 1. **Carga de variables y conexión a MongoDB**
- Usa variables de entorno (`.env`) para conectarse a la base de datos MongoDB.
- Define la latitud y longitud de la vivienda y los días festivos nacionales.

### 2. **Clase `ProcesadorDatos`**
- Centraliza todo el flujo de procesamiento de datos.

#### Métodos principales:
- **`obtener_datos_mongo`**: Descarga datos de una colección de MongoDB y los convierte en DataFrame.
- **`limpiar_ambientes`**: Limpia y normaliza los datos de ambientes (personas, festivo, cielo, etc).
- **`limpiar_periodico`**: Limpia y convierte tipos de datos en la colección principal.
- **`agregar_features_tiempo`**: Añade columnas de fecha/hora (día, mes, hora, minuto, unix).
- **`agregar_features_ciclicas`**: Añade variables cíclicas (seno/coseno de hora, semana, mes, año) para mejorar el aprendizaje del modelo.
- **`combinar_ambientes`**: Une los datos de ambientes con los datos principales.
- **`calcular_posicion_solar`**: Calcula la posición solar (elevación) para cada registro usando `pvlib`.
- **`limpiar_e_imputar`**: Imputa valores nulos, corrige outliers y transforma la columna de personas en variables binarias (Papa, Mama, Ipe, Javi).
- **`rellenar_temp_humedad_por_mes`**: Imputa temperatura y humedad faltantes usando la mediana mensual.
- **`rellenar_festivos`**: Marca los días festivos nacionales en los datos.
- **`obtener_df_limpia`**: Ejecuta todo el pipeline y devuelve un DataFrame limpio listo para entrenar modelos.

### 3. **Ejecución directa**
- Si ejecutas el script directamente, imprime un resumen del DataFrame limpio generado.

---

## 🔄 Flujo general del procesamiento

1. Conexión a MongoDB y descarga de datos.
2. Limpieza y normalización de datos de ambientes y principales.
3. Enriquecimiento con features temporales y cíclicas.
4. Unión de datos de ambientes y principales.
5. Imputación de valores nulos y outliers.
6. Cálculo de posición solar.
7. Exportación de DataFrame limpio.

---

## 🛠️ ¿Cómo modificar el procesamiento?

- **Agregar nuevos features**: Añade métodos en la clase o modifica los existentes (por ejemplo, para nuevas variables meteorológicas).
- **Cambiar la fuente de datos**: Modifica `obtener_datos_mongo` para conectarte a otra colección o base de datos.
- **Ajustar imputaciones**: Cambia la lógica en `rellenar_temp_humedad_por_mes` o `limpiar_e_imputar` según tus necesidades.
- **Añadir más días festivos**: Modifica la lista `FESTIVOS_ES`.

### Ejemplo: añadir una feature de viento
```python
# En ProcesadorDatos.agregar_features_tiempo()
df['viento'] = ... # tu lógica aquí
```

---

## 📦 Dependencias

- pandas, numpy, scikit-learn, pvlib, pymongo, python-dotenv
- (Verifica `requirements.txt` en la raíz del proyecto)

Instala dependencias:
```bash
pip install -r ../requirements.txt
```

---

## 💡 Buenas prácticas
- Mantén una copia de los datos originales.
- Documenta cada paso de limpieza y transformación.
- Usa nombres descriptivos para los archivos generados.
- Revisa los datos antes de entrenar modelos.

---

## 📝 Ejemplo de uso
```python
# Procesamiento de datos
def main():
    procesador = ProcesadorDatos(MONGO_URI, DB_NAME, LATITUD, LONGITUD)
    df_limpio = procesador.obtener_df_limpia()
    df_limpio.to_csv('datos_limpios.csv', index=False)

if __name__ == "__main__":
    main()
```

---

¡La calidad de tus datos es la base de buenas predicciones! 📈🧠 