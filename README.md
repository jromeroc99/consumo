# ⚡️ Consumo Energético - Smart Home Predictor

¡Bienvenido a tu asistente inteligente para predecir el consumo energético de tu vivienda! 🏡✨

---

## 🚀 ¿Qué es este proyecto?

Una solución fullstack que predice el consumo energético de una casa, usando modelos de machine learning y una interfaz moderna, minimalista y responsiva. Ideal para visualizar y optimizar el uso de energía en el hogar.

---

## 🧩 Estructura del Proyecto

```
consumo/
  datos/           # Scripts de procesamiento y limpieza de datos
  entrenamiento/   # Modelos y scripts de entrenamiento ML
  frontend/        # Vite + React + Material UI
  main.py          # API FastAPI principal
  requirements.txt # Dependencias Python
```

---

## 🛠️ Instalación Rápida

### 1. Clona el repositorio
```bash
git clone <repo-url>
cd consumo
```

### 2. Instala dependencias Python y ejecuta la API (FastAPI)
```bash
pip install -r requirements.txt
uvicorn main:app --reload
```
- Accede a la API en: [http://localhost:8000/docs](http://localhost:8000/docs)

### 3. Frontend (Vite + React + Material UI)
```bash
cd frontend
npm install
npm run dev
```
- Abre la app en: [http://localhost:5173](http://localhost:5173)

---

## 🧠 ¿Cómo entrenar y modificar el modelo?

1. **Prepara tus datos**
   - Usa los scripts en `datos/` para limpiar y procesar tus datos.
   - Ejemplo: `python datos/procesamiento.py`

2. **Entrena el modelo**
   - Modifica y ejecuta el código en `entrenamiento/modelo_pytorch.py` para entrenar tu modelo personalizado.
   - Guarda el modelo y los scalers generados.

3. **Integra el modelo entrenado**
   - Asegúrate de que la API (main.py) carga el modelo y los scalers correctos.
   - Reinicia la API si actualizas el modelo.

4. **Predice desde el frontend**
   - Usa la interfaz web para enviar datos y visualizar predicciones en tiempo real.

---

## 🔄 Flujo de trabajo

1. 📥 **Procesa y limpia** tus datos (`datos/`)
2. 🧑‍💻 **Entrena** y ajusta el modelo (`entrenamiento/`)
3. 🧪 **Evalúa** y guarda el modelo entrenado
4. 🚀 **Levanta la API** (`main.py`)
5. 💻 **Abre el frontend** y ¡empieza a predecir!

---

## ✨ Características
- Interfaz web minimalista y elegante
- Predicción de consumo por fases (`fase0`, `fase1`, `fase2`)
- API robusta con FastAPI
- Fácil de modificar y extender
- Código limpio y modular

---

## 📂 Estructura recomendada

- `datos/` — Scripts para procesar y limpiar datos
- `entrenamiento/` — Modelos y entrenamiento ML
- `frontend/` — Interfaz de usuario moderna
- `main.py` — API FastAPI principal
- `requirements.txt` — Dependencias Python

---

## 📝 Notas
- Coloca los archivos del modelo y scalers en la raíz del proyecto o donde los espere la API.
- Puedes personalizar el modelo y la UI según tus necesidades.
- Consulta la documentación de cada carpeta para más detalles.

---

## 🤝 Contribuciones
¡Pull requests y sugerencias son bienvenidas! Siéntete libre de mejorar el proyecto.

---

## 📧 Contacto
¿Dudas o sugerencias? Abre un issue o contacta al autor.

---

_Disfruta optimizando tu consumo energético!_ ⚡️🏡