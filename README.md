# 🚗 End-to-End License Plate Recognition (LPR) Pipeline

Este repositorio contiene un pipeline completo de Machine Learning para el Reconocimiento Óptico de Caracteres (OCR) en placas vehiculares. Abarca desde la generación de datos sintéticos procedurales hasta un script de inferencia robusto que combina detección de objetos (YOLO), regresión heurística (RANSAC) y clasificación profunda (CNN).

El proyecto está diseñado para ser modular, permitiendo entrenar y evaluar cada componente de la arquitectura por separado, y ofrece herramientas experimentales avanzadas (U-Net, Regresión de 5 grados de libertad) para optimizar el rendimiento en condiciones del mundo real.

![Geometry Nodes Setup](PipelineOK.png)
---

## 🏗️ Arquitectura del Pipeline Principal

El flujo de trabajo principal se divide en 4 etapas, cada una manejada por su respectivo script o notebook:

### 1. Generación de Datos Sintéticos
* **Script:** `plate_generator_optimized_2.py`
* **Descripción:** Generador procedural de alto rendimiento que produce imágenes de placas aumentadas (sucias, rotadas, con ruido). 
* **Salidas por placa:** Imagen final, imagen limpia (ground truth visual), máscara binaria de los caracteres, y un archivo `.json` con el *bounding box* exacto de cada letra/número.

### 2. Detección de Caracteres (YOLO)
* **Notebooks:** `dataset_yolo_character.ipynb`
* **Descripción:** Convierte los `.json` generados al formato nativo de YOLO y genera los archivos `.yaml` de configuración. Entrena un modelo YOLO personalizado para detectar y encuadrar cada carácter individual dentro de la imagen de una placa, sin importar el ruido de fondo.

### 3. Clasificación de Caracteres (CNN)
* **Notebooks:** `processDataset.ipynb` y `CNN_characteres.ipynb`
* **Descripción:** `processDataset` recorta cada letra usando las coordenadas del JSON para crear un dataset masivo de caracteres aislados. Luego, la CNN se entrena utilizando un enfoque de **Curriculum Learning**: primero se expone a conjuntos de letras limpias para aprender la topología base, y luego a iteraciones de letras sucias/distorsionadas para mejorar la generalización.

### 4. Inferencia Completa (Pipeline de Producción)
* **Notebook:** `inference_pepeline.ipynb`
* **Descripción:** El script que une todo el sistema. Dada una imagen de entrada:
  1. **YOLO** detecta todos los posibles caracteres y devuelve sus bounding boxes.
  2. **Filtro RANSAC:** Aplica una regresión lineal robusta sobre los centros de las cajas detectadas para encontrar el "renglón principal" de la placa. Si hay múltiples líneas (ruido o placas de dos niveles), selecciona la línea que abarque la mayor área.
  3. **CNN Classifier:** Pasa únicamente los recortes validados por RANSAC a través de la CNN para obtener la cadena de texto final.

---

## 🔬 Módulos Experimentales y de Optimización (Extras)

Además del pipeline base, este repositorio incluye módulos avanzados para abordar casos límite (edge cases) y optimizar el tiempo de inferencia.

### 🎯 Extracción Global de ROI (CNN Regressor)
* **Notebook:** `TEST_box_detector_CNN.ipynb`
* **Descripción:** Entrena una CNN de regresión para predecir 5 valores simultáneos: `[x_center, y_center, width, height, angle]`. 
* **Caso de Uso:** Predice un único *bounding box* global que envuelve todo el texto de la placa (incluso si está rotada). Puede sustituir el paso de RANSAC en la inferencia, aislando la Región de Interés (ROI) de manera mucho más rápida y determinista.

### ✂️ Herramientas de Preprocesamiento de Dataset
* **Scripts:** `processDataset_mask_JSON.py` y `processDataset_denoise_JSON.py`
* **Descripción:** Scripts de utilidad que leen los JSON originales y recortan automáticamente la zona de interés global de la placa, descartando los bordes y el fondo del vehículo para crear datasets más enfocados.

### 🧬 Segmentación U-Net para "Denoising"
* **Notebook:** `U_net_mask.ipynb`
* **Descripción:** Entrena una arquitectura **U-Net** para realizar traducción de imagen a imagen (Image-to-Image translation). Toma una placa sucia/ruidosa y predice una máscara binaria perfectamente limpia de los caracteres.
* **Trabajo Futuro:** Integrar la salida de esta U-Net directamente a la CNN o a YOLO para realizar la inferencia sobre imágenes "limpias" sintéticamente, aumentando drásticamente la precisión.

---

## 🚀 Cómo Empezar (Flujo de Trabajo)

Para replicar este proyecto o entrenar tus propios modelos, sigue este orden:

1. Ejecuta `plate_generator_optimized_2.py` para generar tu dataset base (ej. 50,000 placas).
2. Usa `dataset_yolo_character.ipynb` para preparar y entrenar el modelo detector de cajas.
3. Ejecuta `processDataset.ipynb` para extraer los caracteres individuales en carpetas separadas por clase.
4. Entrena el clasificador de letras en `CNN_characteres.ipynb`.
5. Abre `inference_pepeline.ipynb`, carga tus pesos entrenados (YOLO `.pt` y CNN `.pth`/`.h5`) y prueba el sistema con imágenes del mundo real.

## 🛠️ Requisitos Técnicos
* PyTorch / TensorFlow (Dependiendo del framework usado para las CNN/U-Net)
* Ultralytics (YOLOv8 o similar)
* OpenCV, Scikit-Learn (Para RANSAC), Pandas, Numpy.
