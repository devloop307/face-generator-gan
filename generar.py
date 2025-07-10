import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import argparse
from datetime import datetime
import random

class GeneradorRostros:
    """Clase para generar rostros usando un modelo GAN pre-entrenado"""
    
    def __init__(self, model_path, latent_dim=100):
        self.latent_dim = latent_dim
        self.model_path = model_path
        self.generator = None
        self.cargar_modelo()
    
    def cargar_modelo(self):
        """Cargar el modelo generador pre-entrenado"""
        try:
            self.generator = tf.keras.models.load_model(self.model_path)
            print(f"✅ Modelo cargado exitosamente desde: {self.model_path}")
            print(f"📊 Dimensión de entrada: {self.generator.input_shape}")
            print(f"🖼️ Dimensión de salida: {self.generator.output_shape}")
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            raise
    
    def generar_rostros(self, num_rostros=1, semilla=None):
        """Generar rostros aleatorios"""
        if semilla is not None:
            np.random.seed(semilla)
            tf.random.set_seed(semilla)
        
        # Generar ruido aleatorio
        noise = tf.random.normal([num_rostros, self.latent_dim])
        
        # Generar imágenes
        generated_images = self.generator(noise, training=False)
        
        # Desnormalizar de [-1, 1] a [0, 1]
        generated_images = (generated_images + 1) / 2.0
        
        # Convertir a numpy y asegurar rango válido
        generated_images = tf.clip_by_value(generated_images, 0.0, 1.0)
        
        return generated_images.numpy()
    
    def generar_con_interpolacion(self, num_pasos=10, semilla1=None, semilla2=None):
        """Generar rostros con interpolación entre dos puntos"""
        if semilla1 is not None:
            np.random.seed(semilla1)
            noise1 = tf.random.normal([1, self.latent_dim])
        else:
            noise1 = tf.random.normal([1, self.latent_dim])
        
        if semilla2 is not None:
            np.random.seed(semilla2)
            noise2 = tf.random.normal([1, self.latent_dim])
        else:
            noise2 = tf.random.normal([1, self.latent_dim])
        
        # Crear interpolación lineal
        interpolated_images = []
        for i in range(num_pasos):
            alpha = i / (num_pasos - 1)
            interpolated_noise = noise1 * (1 - alpha) + noise2 * alpha
            
            generated_image = self.generator(interpolated_noise, training=False)
            generated_image = (generated_image + 1) / 2.0
            generated_image = tf.clip_by_value(generated_image, 0.0, 1.0)
            
            interpolated_images.append(generated_image.numpy()[0])
        
        return np.array(interpolated_images)
    
    def generar_variaciones(self, semilla_base, num_variaciones=9, factor_variacion=0.1):
        """Generar variaciones de un rostro base"""
        np.random.seed(semilla_base)
        noise_base = tf.random.normal([1, self.latent_dim])
        
        variaciones = []
        for i in range(num_variaciones):
            # Añadir ruido pequeño para crear variaciones
            variation_noise = tf.random.normal([1, self.latent_dim]) * factor_variacion
            varied_noise = noise_base + variation_noise
            
            generated_image = self.generator(varied_noise, training=False)
            generated_image = (generated_image + 1) / 2.0
            generated_image = tf.clip_by_value(generated_image, 0.0, 1.0)
            
            variaciones.append(generated_image.numpy()[0])
        
        return np.array(variaciones)
    
    def guardar_rostros(self, images, output_dir="rostros_generados", prefix="rostro"):
        """Guardar rostros generados como imágenes individuales"""
        os.makedirs(output_dir, exist_ok=True)
        
        saved_files = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for i, img in enumerate(images):
            # Convertir a formato PIL
            img_array = (img * 255).astype(np.uint8)
            pil_image = Image.fromarray(img_array)
            
            # Guardar imagen
            filename = f"{prefix}_{timestamp}_{i+1:03d}.png"
            filepath = os.path.join(output_dir, filename)
            pil_image.save(filepath)
            saved_files.append(filepath)
        
        print(f"💾 {len(images)} rostros guardados en: {output_dir}")
        return saved_files
    
    def crear_mosaico(self, images, filas=None, columnas=None, titulo="Rostros Generados"):
        """Crear un mosaico de rostros generados"""
        num_images = len(images)
        
        if filas is None and columnas is None:
            # Calcular dimensiones automáticamente
            filas = int(np.ceil(np.sqrt(num_images)))
            columnas = int(np.ceil(num_images / filas))
        elif filas is None:
            filas = int(np.ceil(num_images / columnas))
        elif columnas is None:
            columnas = int(np.ceil(num_images / filas))
        
        fig, axes = plt.subplots(filas, columnas, figsize=(columnas * 3, filas * 3))
        fig.suptitle(titulo, fontsize=16)
        
        if filas == 1 and columnas == 1:
            axes = [axes]
        elif filas == 1 or columnas == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for i in range(filas * columnas):
            if i < num_images:
                axes[i].imshow(images[i])
                axes[i].set_title(f'Rostro {i+1}')
            axes[i].axis('off')
        
        plt.tight_layout()
        return fig
    
    def crear_animacion_interpolacion(self, images, output_path="interpolacion.gif", duration=200):
        """Crear GIF animado de interpolación"""
        pil_images = []
        for img in images:
            img_array = (img * 255).astype(np.uint8)
            pil_image = Image.fromarray(img_array)
            pil_images.append(pil_image)
        
        # Guardar como GIF
        pil_images[0].save(
            output_path,
            save_all=True,
            append_images=pil_images[1:],
            duration=duration,
            loop=0
        )
        
        print(f"🎬 Animación guardada en: {output_path}")
    
    def analizar_diversidad(self, num_muestras=100):
        """Analizar la diversidad de rostros generados"""
        print(f"📊 Analizando diversidad con {num_muestras} muestras...")
        
        # Generar muestras
        rostros = self.generar_rostros(num_muestras)
        
        # Calcular estadísticas básicas
        mean_pixel = np.mean(rostros)
        std_pixel = np.std(rostros)
        
        # Calcular diferencias entre imágenes
        diferencias = []
        for i in range(len(rostros)):
            for j in range(i + 1, len(rostros)):
                diff = np.mean(np.abs(rostros[i] - rostros[j]))
                diferencias.append(diff)
        
        diversidad_promedio = np.mean(diferencias)
        
        print(f"📈 Resultados del análisis:")
        print(f"  - Pixel promedio: {mean_pixel:.4f}")
        print(f"  - Desviación estándar: {std_pixel:.4f}")
        print(f"  - Diversidad promedio: {diversidad_promedio:.4f}")
        
        return {
            'mean_pixel': mean_pixel,
            'std_pixel': std_pixel,
            'diversidad_promedio': diversidad_promedio,
            'num_muestras': num_muestras
        }


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description='Generar rostros usando GAN pre-entrenado')
    parser.add_argument('--model_path', type=str, required=True, help='Ruta al modelo generador')
    parser.add_argument('--num_rostros', type=int, default=16, help='Número de rostros a generar')
    parser.add_argument('--output_dir', type=str, default='rostros_generados', help='Directorio de salida')
    parser.add_argument('--semilla', type=int, default=None, help='Semilla para reproducibilidad')
    parser.add_argument('--modo', type=str, default='normal', 
                       choices=['normal', 'interpolacion', 'variaciones', 'diversidad'],
                       help='Modo de generación')
    parser.add_argument('--mostrar_mosaico', action='store_true', help='Mostrar mosaico de rostros')
    parser.add_argument('--guardar_individual', action='store_true', help='Guardar rostros individuales')
    parser.add_argument('--latent_dim', type=int, default=100, help='Dimensión del espacio latente')
    
    args = parser.parse_args()
    
    # Verificar que existe el modelo
    if not os.path.exists(args.model_path):
        print(f"❌ Error: El modelo {args.model_path} no existe")
        return
    
    print("🎨 Iniciando generador de rostros")
    print(f"📊 Parámetros:")
    print(f"  - Modelo: {args.model_path}")
    print(f"  - Número de rostros: {args.num_rostros}")
    print(f"  - Modo: {args.modo}")
    print(f"  - Directorio de salida: {args.output_dir}")
    if args.semilla:
        print(f"  - Semilla: {args.semilla}")
    
    # Crear instancia del generador
    generador = GeneradorRostros(args.model_path, args.latent_dim)
    
    # Generar rostros según el modo
    if args.modo == 'normal':
        rostros = generador.generar_rostros(args.num_rostros, args.semilla)
        titulo = f"Rostros Generados ({args.num_rostros})"
        
    elif args.modo == 'interpolacion':
        rostros = generador.generar_con_interpolacion(args.num_rostros, args.semilla)
        titulo = f"Interpolación de Rostros ({args.num_rostros} pasos)"
        
        # Crear animación GIF
        generador.crear_animacion_interpolacion(rostros, 
                                              f"{args.output_dir}/interpolacion.gif")
        
    elif args.modo == 'variaciones':
        semilla_base = args.semilla if args.semilla else random.randint(1, 10000)
        rostros = generador.generar_variaciones(semilla_base, args.num_rostros)
        titulo = f"Variaciones de Rostro (Semilla: {semilla_base})"
        
    elif args.modo == 'diversidad':
        stats = generador.analizar_diversidad(args.num_rostros)
        print("✅ Análisis de diversidad completado")
        return
    
    # Guardar rostros individuales si se solicita
    if args.guardar_individual:
        generador.guardar_rostros(rostros, args.output_dir)
    
    # Mostrar mosaico si se solicita
    if args.mostrar_mosaico:
        fig = generador.crear_mosaico(rostros, titulo=titulo)
        
        # Guardar mosaico
        os.makedirs(args.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mosaico_path = f"{args.output_dir}/mosaico_{timestamp}.png"
        fig.savefig(mosaico_path, dpi=150, bbox_inches='tight')
        print(f"🖼️ Mosaico guardado en: {mosaico_path}")
        
        plt.show()
    
    print("🎉 ¡Generación completada exitosamente!")


if __name__ == "__main__":
    main()