import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import argparse
from pathlib import Path

class GeneradorRostros:
    """Clase para crear y entrenar un GAN generador de rostros"""
    def __init__(self, latent_dim=100, img_shape=(64, 64, 3)):
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.img_height, self.img_width, self.channels = img_shape
        
        # Crear modelos
        self.generator = self.crear_generador()
        self.discriminator = self.crear_discriminador()
        
        # Compilar discriminador
        self.discriminator.compile(
            loss='binary_crossentropy',
            optimizer=tf.keras.optimizers.Adam(0.0002, 0.5),
            metrics=['accuracy']
        )
        
        # Para el modelo combinado, solo entrenar el generador
        self.discriminator.trainable = False
        
        # Crear modelo combinado
        noise = tf.keras.Input(shape=(self.latent_dim,))
        img = self.generator(noise)
        validity = self.discriminator(img)
        
        self.combined = tf.keras.Model(noise, validity)
        self.combined.compile(
            loss='binary_crossentropy',
            optimizer=tf.keras.optimizers.Adam(0.0002, 0.5)
        )
        
        print("✅ Modelos creados exitosamente")
        print(f"📊 Dimensión latente: {self.latent_dim}")
        print(f"🖼️ Forma de imagen: {self.img_shape}")
    
    def crear_generador(self):
        """Crear el modelo generador"""
        model = tf.keras.Sequential()
        
        # Capa densa inicial
        model.add(layers.Dense(8 * 8 * 256, use_bias=False, input_shape=(self.latent_dim,)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        
        # Reshape a 3D
        model.add(layers.Reshape((8, 8, 256)))
        
        # Upsample a 16x16
        model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        
        # Upsample a 32x32
        model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        
        # Upsample a 64x64
        model.add(layers.Conv2DTranspose(self.channels, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        
        return model
    
    def crear_discriminador(self):
        """Crear el modelo discriminador"""
        model = tf.keras.Sequential()
        
        # Capa convolucional 1
        model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=self.img_shape))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))
        
        # Capa convolucional 2
        model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))
        
        # Capa convolucional 3
        model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))
        
        # Flatten y clasificación
        model.add(layers.Flatten())
        model.add(layers.Dense(1, activation='sigmoid'))
        
        return model
    
    def cargar_datos(self, data_path, batch_size=32):
        """Cargar y preprocesar el dataset"""
        print(f"📁 Cargando datos desde: {data_path}")
        
        # Crear dataset desde directorio
        dataset = tf.keras.utils.image_dataset_from_directory(
            data_path,
            labels=None,
            image_size=(self.img_height, self.img_width),
            batch_size=batch_size,
            shuffle=True
        )
        
        # Normalizar imágenes a [-1, 1]
        dataset = dataset.map(lambda x: (x - 127.5) / 127.5)
        
        print(f"✅ Dataset cargado exitosamente")
        return dataset
    
    def generar_muestras(self, num_samples=16):
        """Generar muestras del generador"""
        noise = tf.random.normal([num_samples, self.latent_dim])
        generated_images = self.generator(noise, training=False)
        
        # Desnormalizar imágenes
        generated_images = (generated_images + 1) / 2.0
        
        return generated_images
    
    def guardar_muestras(self, epoch, samples_dir="muestras"):
        """Guardar muestras generadas"""
        os.makedirs(samples_dir, exist_ok=True)
        
        generated_images = self.generar_muestras(16)
        
        fig, axes = plt.subplots(4, 4, figsize=(10, 10))
        fig.suptitle(f'Época {epoch}', fontsize=16)
        
        for i in range(16):
            row = i // 4
            col = i % 4
            axes[row, col].imshow(generated_images[i])
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{samples_dir}/epoch_{epoch:04d}.png")
        plt.close()
    
    def entrenar(self, dataset, epochs=100, sample_interval=10, save_interval=50):
        """Entrenar el GAN"""
        print(f"🚀 Iniciando entrenamiento por {epochs} épocas")
        
        # Listas para almacenar pérdidas
        d_losses = []
        g_losses = []
        
        start_time = datetime.now()
        
        for epoch in range(epochs):
            epoch_d_loss = []
            epoch_g_loss = []
            
            for batch in dataset:
                batch_size = tf.shape(batch)[0]
                # ---------------------
                # Entrenar Discriminador
                # ---------------------        
                # Generar ruido aleatorio
                noise = tf.random.normal([batch_size, self.latent_dim])
                
                # Generar imágenes falsas
                generated_images = self.generator(noise)
                
                # Etiquetas para imágenes reales y falsas
                real_labels = tf.ones((batch_size, 1))
                fake_labels = tf.zeros((batch_size, 1))
                
                # Entrenar discriminador con imágenes reales
                d_loss_real = self.discriminator.train_on_batch(batch, real_labels)
                
                # Entrenar discriminador con imágenes falsas
                d_loss_fake = self.discriminator.train_on_batch(generated_images, fake_labels)
                
                # Pérdida total del discriminador
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                
                # ---------------------
                # Entrenar Generador
                # ---------------------
                
                # Generar ruido aleatorio
                noise = tf.random.normal([batch_size, self.latent_dim])
                
                # Etiquetas para engañar al discriminador
                misleading_labels = tf.ones((batch_size, 1))
                
                # Entrenar generador
                g_loss = self.combined.train_on_batch(noise, misleading_labels)
                
                epoch_d_loss.append(d_loss[0])
                epoch_g_loss.append(g_loss)
            
            # Calcular pérdidas promedio de la época
            avg_d_loss = np.mean(epoch_d_loss)
            avg_g_loss = np.mean(epoch_g_loss)
            
            d_losses.append(avg_d_loss)
            g_losses.append(avg_g_loss)
            
            # Mostrar progreso
            if epoch % sample_interval == 0:
                elapsed_time = datetime.now() - start_time
                print(f"Época {epoch:04d}/{epochs} - "
                      f"D loss: {avg_d_loss:.4f} - "
                      f"G loss: {avg_g_loss:.4f} - "
                      f"Tiempo: {elapsed_time}")
                
                # Guardar muestras
                self.guardar_muestras(epoch)
            
            # Guardar modelos
            if epoch % save_interval == 0 and epoch > 0:
                self.guardar_modelos(epoch)
        
        print("✅ Entrenamiento completado")
        
        # Guardar gráfico de pérdidas
        self.graficar_perdidas(d_losses, g_losses)
        
        return d_losses, g_losses
    
    def graficar_perdidas(self, d_losses, g_losses):
        """Graficar las pérdidas del entrenamiento"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(d_losses, label='Discriminador')
        plt.plot(g_losses, label='Generador')
        plt.title('Pérdidas durante el entrenamiento')
        plt.xlabel('Época')
        plt.ylabel('Pérdida')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(d_losses, label='Discriminador')
        plt.title('Pérdida del Discriminador')
        plt.xlabel('Época')
        plt.ylabel('Pérdida')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('perdidas_entrenamiento.png')
        plt.show()
    
    def guardar_modelos(self, epoch):
        """Guardar los modelos"""
        os.makedirs('modelos', exist_ok=True)
        
        self.generator.save(f'modelos/generador_epoch_{epoch}.h5')
        self.discriminator.save(f'modelos/discriminador_epoch_{epoch}.h5')
        
        print(f"💾 Modelos guardados en época {epoch}")
    
    def cargar_modelos(self, generator_path, discriminator_path):
        """Cargar modelos pre-entrenados"""
        self.generator = tf.keras.models.load_model(generator_path)
        self.discriminator = tf.keras.models.load_model(discriminator_path)
        print("✅ Modelos cargados exitosamente")


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description='Entrenamiento GAN para generación de rostros')
    parser.add_argument('--data_path', type=str, required=True, help='Ruta al dataset')
    parser.add_argument('--epochs', type=int, default=100, help='Número de épocas')
    parser.add_argument('--batch_size', type=int, default=32, help='Tamaño del batch')
    parser.add_argument('--latent_dim', type=int, default=100, help='Dimensión del espacio latente')
    parser.add_argument('--sample_interval', type=int, default=10, help='Intervalo para guardar muestras')
    parser.add_argument('--save_interval', type=int, default=50, help='Intervalo para guardar modelos')
    
    args = parser.parse_args()
    
    # Verificar que existe el directorio de datos
    if not os.path.exists(args.data_path):
        print(f"❌ Error: El directorio {args.data_path} no existe")
        return
    
    print("🎯 Iniciando aplicación GAN para generación de rostros")
    print(f"📊 Parámetros:")
    print(f"  - Dataset: {args.data_path}")
    print(f"  - Épocas: {args.epochs}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Dimensión latente: {args.latent_dim}")
    
    # Crear instancia del generador
    gan = GeneradorRostros(latent_dim=args.latent_dim)
    
    # Cargar datos
    dataset = gan.cargar_datos(args.data_path, args.batch_size)
    
    # Entrenar el modelo
    d_losses, g_losses = gan.entrenar(
        dataset=dataset,
        epochs=args.epochs,
        sample_interval=args.sample_interval,
        save_interval=args.save_interval
    )
    
    # Generar muestras finales
    print("🎨 Generando muestras finales...")
    gan.guardar_muestras(args.epochs, "muestras_finales")
    
    # Guardar modelos finales
    gan.guardar_modelos("final")
    
    print("🎉 ¡Entrenamiento completado exitosamente!")


if __name__ == "__main__":
    # Configurar GPU si está disponible
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        print(f"🚀 GPU detectada: {len(gpus)} dispositivo(s)")
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"⚠️ Error configurando GPU: {e}")
    else:
        print("💻 Usando CPU para entrenamiento")
    
    main()