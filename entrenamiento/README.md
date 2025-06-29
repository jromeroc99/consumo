# 🤖 Entrenamiento de Modelos - Consumo Energético

Esta carpeta contiene los scripts y utilidades para entrenar, evaluar y guardar modelos de predicción de consumo energético. Aquí puedes experimentar, ajustar y mejorar los modelos de machine learning usados por la API.

---

## 📂 Estructura

- `modelo_pytorch.py` — Script principal para definir, entrenar y guardar modelos con PyTorch
- `__init__.py` — Inicialización del módulo

---

## ⏰ ¿Qué pasa con la variable `tiempo` (datetime)?

- La columna `tiempo` es un campo de tipo datetime que representa la marca temporal de cada registro.
- **No se usa directamente como input del modelo**. En su lugar, durante el procesamiento de datos (`procesamiento.py`), se generan variables derivadas de `tiempo`:
  - Features temporales: día, mes, hora, minuto, unix timestamp.
  - Features cíclicas: seno/coseno de hora, semana, mes, año (para capturar periodicidad y estacionalidad).
- Cuando se entrena el modelo (`modelo_pytorch.py`), **solo se usan las variables numéricas** resultantes del procesamiento. La columna original `tiempo` no se incluye como input.
- Esto permite que el modelo aprenda patrones temporales sin depender de la fecha/hora exacta, sino de sus componentes y ciclos.

---

## 🚦 ¿Qué hace `modelo_pytorch.py`? (Paso a paso)

### 1. **Carga y preparación de datos**
- Usa el procesador de datos (`ProcesadorDatos`) para obtener un DataFrame limpio desde MongoDB o un CSV.
- Separa las variables objetivo (`fase0`, `fase1`, `fase2`) y las variables de entrada.
- Elimina columnas no numéricas (incluida la columna `tiempo`).

### 2. **Preprocesamiento**
- Escala los datos de entrada y salida con `StandardScaler` para mejorar el aprendizaje.
- Divide los datos en entrenamiento y test (80/20).

### 3. **Dataset y DataLoader**
- Define la clase `ConsumoDataset` para usar los datos con PyTorch.
- Usa `DataLoader` para cargar los datos en batches durante el entrenamiento y test.

### 4. **Definición del modelo**
- Define una red neuronal MLP (perceptrón multicapa) con dos capas ocultas (128 y 64 neuronas, activación ReLU).
- La salida tiene 3 neuronas (una por cada fase de consumo).

### 5. **Entrenamiento**
- Usa Adam como optimizador y MSE como función de pérdida.
- Entrena el modelo durante N épocas, mostrando la pérdida en cada época.

### 6. **Evaluación**
- Calcula la pérdida en el conjunto de test tras el entrenamiento.

### 7. **Guardado**
- Guarda el modelo entrenado (`modelo_consumo.pth`) y los scalers (`scaler_X.pkl`, `scaler_y.pkl`).

---

## 🛠️ ¿Cómo modificar el modelo o el entrenamiento?

- **Cambiar arquitectura**: Modifica la clase `MLP` para añadir/quitar capas, cambiar activaciones, etc.
- **Ajustar hiperparámetros**: Cambia `BATCH_SIZE`, `EPOCHS`, `LEARNING_RATE` al inicio del script.
- **Probar otros modelos**: Sustituye la clase `MLP` por otra arquitectura (por ejemplo, una red más profunda, regularización, dropout, etc).
- **Cambiar la función de pérdida**: Usa MAE, Huber, etc. en vez de MSE.
- **Añadir callbacks o early stopping**: Implementa lógica adicional en el bucle de entrenamiento.

### Ejemplo: añadir una capa y dropout
```python
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
```

---

## 📦 Dependencias

- PyTorch
- pandas, numpy, scikit-learn
- (Verifica `requirements.txt` en la raíz del proyecto)

Instala dependencias:
```bash
pip install -r ../requirements.txt
```

---

## 💡 Buenas prácticas
- Versiona tus modelos y scalers (usa nombres descriptivos o carpetas por fecha/experimento).
- Documenta los cambios y resultados de cada experimento.
- Mantén el código modular y limpio para facilitar mejoras.
- Prueba diferentes arquitecturas y compara resultados.

---

## 📝 Ejemplo de uso
```bash
# Entrenamiento estándar
python entrenamiento/modelo_pytorch.py

# Modifica hiperparámetros o arquitectura dentro del script
# Guarda el modelo y los scalers generados
```

---

¡Experimenta, aprende y mejora tus predicciones! 🚀📊 