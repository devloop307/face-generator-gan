# Generador de Rostros GAN

Implementación en Python de GAN para generar rostros humanos realistas usando aprendizaje profundo.

## 🎯 Descripción del Proyecto

Este proyecto implementa una Red Generativa Adversarial (GAN) para generar rostros humanos sintéticos. El modelo aprende de un conjunto de datos de imágenes de rostros reales y crea nuevos rostros de aspecto realista que no corresponden a ninguna persona real.

## 🚀 Características

- **Arquitectura de Aprendizaje Profundo**: Usa GAN con redes Generador y Discriminador
- **Salida de Alta Calidad**: Genera rostros humanos realistas
- **Personalizable**: Parámetros ajustables para diferentes resultados
- **Visualización de Entrenamiento**: Seguimiento de pérdidas en tiempo real y generación de muestras
- **Fácil de Usar**: Interfaz simple para generar nuevos rostros

## 📋 Requisitos

- Python 3.8+
- TensorFlow 2.x o PyTorch
- NumPy
- Matplotlib
- PIL (Pillow)
- OpenCV

## 🛠️ Instalación

1. Clona el repositorio:
```bash
git clone https://github.com/tuusuario/generador-rostros-gan.git
cd generador-rostros-gan
```

2. Crea un entorno virtual:
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. Instala las dependencias:
```bash
pip install -r requirements.txt
```

## 📊 Conjunto de Datos

Este proyecto utiliza conjuntos de datos de rostros como:
- CelebA (Rostros de Celebridades)
- FFHQ (Flickr-Faces-HQ)
- Conjuntos de datos personalizados de rostros

Coloca tu conjunto de datos en el directorio `data/` siguiendo esta estructura:
```
data/
├── train/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
└── test/
    ├── img1.jpg
    └── ...
```

## 🔧 Uso

### Entrenar el Modelo

```bash
python entrenar.py --epochs 100 --batch_size 32 --learning_rate 0.0002
```

### Generar Nuevos Rostros

```bash
python generar.py --num_images 10 --output_dir ./rostros_generados/
```

### Evaluar el Modelo

```bash
python evaluar.py --model_path ./modelos/generador.h5
```

## 📁 Estructura del Proyecto

```
generador-rostros-gan/
├── data/                   # Directorio de datos
├── modelos/                # Modelos guardados
├── rostros_generados/      # Imágenes generadas
├── notebooks/              # Jupyter notebooks
├── src/                    # Código fuente
│   ├── __init__.py
│   ├── modelo_gan.py      # Arquitectura GAN
│   ├── cargador_datos.py  # Preprocesamiento de datos
│   ├── entrenamiento.py   # Lógica de entrenamiento
│   └── utilidades.py      # Funciones utilitarias
├── config/                 # Archivos de configuración
│   └── config.yaml
├── requirements.txt        # Dependencias
├── entrenar.py            # Script de entrenamiento
├── generar.py             # Script de generación
├── evaluar.py             # Script de evaluación
└── README.md              # Este archivo
```

## ⚙️ Configuración

Edita `config/config.yaml` para personalizar:
- Tamaño de imagen y canales
- Arquitectura de red
- Parámetros de entrenamiento
- Funciones de pérdida
- Configuraciones de optimización

## 📈 Resultados

El modelo genera rostros sintéticos de alta calidad después del entrenamiento. Ejemplos de rostros generados:

![Rostros Generados](assets/muestras_generadas.png)

## 🔍 Arquitectura del Modelo

- **Generador**: Transforma ruido aleatorio en imágenes de rostros realistas
- **Discriminador**: Distingue entre rostros reales y generados
- **Función de Pérdida**: Pérdida adversarial con pérdida perceptual opcional
- **Optimización**: Optimizador Adam con programación de tasa de aprendizaje

## 📊 Métricas de Rendimiento

- **Puntuación FID**: Mide la calidad de las imágenes generadas
- **Puntuación IS**: Inception Score para evaluación de diversidad
- **Pérdida de Entrenamiento**: Curvas de pérdida del Generador y Discriminador

## 🤝 Contribuciones

1. Haz fork del repositorio
2. Crea una rama de característica (`git checkout -b feature/caracteristica-increible`)
3. Confirma tus cambios (`git commit -m 'Añadir característica increíble'`)
4. Empuja a la rama (`git push origin feature/caracteristica-increible`)
5. Abre un Pull Request

## 📧 Contacto

Tu Nombre - [tu.email@ejemplo.com](mailto:tu.email@ejemplo.com)

Enlace del Proyecto: [https://github.com/tuusuario/generador-rostros-gan](https://github.com/tuusuario/generador-rostros-gan)

---
