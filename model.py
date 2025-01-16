import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from util import OneHot
import matplotlib.pyplot as plt



#Define generator and discriminator network
class Generator(keras.Model):
    def __init__(self, batch_size, dim_y, dim_z, dim_W1, dim_W2, dim_W3, dim_channel, initializer):
        super().__init__(name='Generator')

        #Parameters
        self.batch_size = batch_size
        self.dim_y = dim_y
        self.dim_z = dim_z
        self.dim_W1 = dim_W1
        self.dim_W2 = dim_W2
        self.dim_W3 = dim_W3
        self.dim_channel = dim_channel
        
        #Layers
        self.layer1_dense = keras.layers.Dense(units=dim_W1,
                                               use_bias=False,
                                               kernel_initializer=initializer)
        self.layer1_batchnorm = keras.layers.BatchNormalization(epsilon=1e-8,
                                                                beta_initializer=initializer,
                                                                gamma_initializer=initializer)
        self.layer1_activation = keras.layers.ReLU()

        self.layer2_dense = keras.layers.Dense(units=dim_W2*6*6,
                                               use_bias=False,
                                               kernel_initializer=initializer)
        self.layer2_batchnorm = keras.layers.BatchNormalization(epsilon=1e-8,
                                                                beta_initializer=initializer,
                                                                gamma_initializer=initializer)
        self.layer2_activation = keras.layers.ReLU()

        self.layer3_conv = keras.layers.Conv2DTranspose(filters=dim_W3,
                                                        kernel_size=5,
                                                        strides=(2,2),
                                                        padding='same',
                                                        kernel_initializer=initializer,
                                                        bias_initializer=initializer)
        self.layer3_batchnorm = keras.layers.BatchNormalization(epsilon=1e-8,
                                                                beta_initializer=initializer,
                                                                gamma_initializer=initializer)
        self.layer3_activation = keras.layers.ReLU()

        self.layer4_conv = keras.layers.Conv2DTranspose(filters=dim_channel,
                                                        kernel_size=5,
                                                        strides=(2,2),
                                                        padding='same',
                                                        kernel_initializer=initializer,
                                                        bias_initializer=initializer)
        self.layer4_batchnorm = keras.layers.BatchNormalization(epsilon=1e-8,
                                                                beta_initializer=initializer,
                                                                gamma_initializer=initializer)

    def call(self, z, y, training: bool = True):
        yb = tf.reshape(y, [self.batch_size, 1, 1, self.dim_y])
        z = tf.concat([z, y], -1)
        h1 = self.layer1_activation(self.layer1_batchnorm(self.layer1_dense(z), training=training))
        h1 = tf.concat([h1, y], -1)
        h2 = self.layer2_activation(self.layer2_batchnorm(self.layer2_dense(h1), training=training))
        h2 = tf.reshape(h2, [self.batch_size, 6, 6, self.dim_W2])
        h2 = tf.concat([h2, yb*tf.ones([self.batch_size, 6, 6, self.dim_y])], -1)
        h3 = self.layer3_activation(self.layer3_batchnorm(self.layer3_conv(h2), training=training))
        h3 = tf.concat([h3, yb*tf.ones([self.batch_size, 12, 12, self.dim_y])], -1)
        h4 = self.layer4_batchnorm(self.layer4_conv(h3), training=training)

        return h4

class Discriminator(keras.Model):
    def __init__(self, batch_size, dim_y, dim_z, dim_W1, dim_W2, dim_W3, dim_channel, initializer):
        super().__init__(name='Discriminator')

        #Parameters
        self.batch_size = batch_size
        self.dim_y = dim_y
        self.dim_z = dim_z
        self.dim_W1 = dim_W1
        self.dim_W2 = dim_W2
        self.dim_W3 = dim_W3
        self.dim_channel = dim_channel
        
        #Layers
        self.layer1_conv = keras.layers.Conv2D(filters=dim_W3,
                                               kernel_size=5,
                                               strides=(2,2),
                                               padding='same',
                                               kernel_initializer=initializer,
                                               bias_initializer=initializer)
        self.layer1_activation = keras.layers.LeakyReLU(negative_slope=0.2)

        self.layer2_conv = keras.layers.Conv2D(filters=dim_W2,
                                               kernel_size=5,
                                               strides=(2,2),
                                               padding='same',
                                               kernel_initializer=initializer,
                                               bias_initializer=initializer)
        self.layer2_batchnorm = keras.layers.BatchNormalization(epsilon=1e-8,
                                                                beta_initializer=initializer,
                                                                gamma_initializer=initializer)
        self.layer2_activation = keras.layers.LeakyReLU(negative_slope=0.2)

        self.layer3_dense = keras.layers.Dense(units=dim_W1, use_bias=False, kernel_initializer=initializer)
        self.layer3_batchnorm = keras.layers.BatchNormalization(epsilon=1e-8,
                                                                beta_initializer=initializer,
                                                                gamma_initializer=initializer)
        self.layer3_activation = keras.layers.LeakyReLU(negative_slope=0.2)

    def call(self, image, y, training: bool = True):
        yb = tf.reshape(y, [self.batch_size, 1, 1, self.dim_y])
        x = tf.concat([image, yb*tf.ones([self.batch_size, 24, 24, self.dim_y])], -1)
        h1 = self.layer1_activation(self.layer1_conv(x))
        h1 = tf.concat([h1, yb*tf.ones([self.batch_size, 12, 12, self.dim_y])], -1)
        h2 = self.layer2_activation(self.layer2_batchnorm(self.layer2_conv(h1), training=training))
        h2 = tf.reshape(h2, [self.batch_size, -1])
        h2 = tf.concat([h2, y], -1)
        h3 = self.layer3_activation(self.layer3_batchnorm(self.layer3_dense(h2), training=training))

        return h3

#Define loss functions
@tf.function
def generator_cost(raw_gen2):
    return -tf.math.reduce_mean(raw_gen2)

@tf.function
def discriminator_cost(raw_real2, raw_gen2):
    return tf.math.reduce_sum(raw_gen2) - tf.math.reduce_sum(raw_real2)

#Define GAN architecture
class GAN():
    def __init__(self,
                 epochs: int = 1000,
                 batch_size: int = 32,
                 image_shape: list = [24, 24, 1],
                 dim_y: int = 6,
                 dim_z: int = 100,
                 dim_W1:int = 1024,
                 dim_W2: int = 128,
                 dim_W3: int = 64,
                 dim_channel: int = 1,
                 learning_rate: float = 1e-4):

        #Parameters
        self.epochs = epochs
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.dim_y = dim_y
        self.dim_z = dim_z
        self.dim_W1 = dim_W1
        self.dim_W2 = dim_W2
        self.dim_W3 = dim_W3
        self.dim_channel = dim_channel
        self.learning_rate = learning_rate
        self.normal = (0, 0.1)

        #Initialization of weights
        initializer = keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=42)

        #Instantiate generator and discriminator network
        self.generator = Generator(batch_size, dim_y, dim_z, dim_W1, dim_W2, dim_W3, dim_channel, initializer)
        self.discriminator = Discriminator(batch_size, dim_y, dim_z, dim_W1, dim_W2, dim_W3, dim_channel, initializer)     
           
        #Optimizers
        self.optimizer_g = keras.optimizers.RMSprop(learning_rate=learning_rate)
        self.optimizer_d = keras.optimizers.RMSprop(learning_rate=learning_rate)

        #Loss functions
        self.generator_loss = generator_cost
        self.discriminator_loss = discriminator_cost

        #Auxiliary
        self.fitting_time = None
    
    def fit(self, x, y):
        self.fitting_time = time.time()
        iterations = 0

        #Control balance of training discriminator vs generator; default k = 4
        k = 4

        #Define generator train step
        @tf.function
        def train_step_g(xs, ys, zs):
            with tf.GradientTape() as tape:
                h4 = self.generator(zs, ys)
                image_gen = keras.ops.sigmoid(h4)
                raw_gen2 = self.discriminator(image_gen, ys)
                p_gen_val = tf.math.reduce_mean(raw_gen2)
            
                gen_loss_val = self.generator_loss(raw_gen2)

                raw_real2 = self.discriminator(xs, ys)
                p_real_val = tf.math.reduce_mean(raw_real2)

                discrim_loss_val = self.discriminator_loss(raw_real2, raw_gen2)
            
            grad_g = tape.gradient(gen_loss_val, self.generator.trainable_variables)
            self.optimizer_g.apply_gradients(zip(grad_g, self.generator.trainable_variables))

            return p_gen_val, p_real_val, discrim_loss_val, gen_loss_val
        
        #Define discriminator train step
        @tf.function
        def train_step_d(xs, ys, zs):
            with tf.GradientTape() as tape:
                h4 = self.generator(zs, ys)
                image_gen = keras.ops.sigmoid(h4)
                raw_gen2 = self.discriminator(image_gen, ys)
                p_gen_val = tf.math.reduce_mean(raw_gen2)

                gen_loss_val = self.generator_loss(raw_gen2)

                raw_real2 = self.discriminator(xs, ys)
                p_real_val = tf.math.reduce_mean(raw_real2)

                discrim_loss_val = self.discriminator_loss(raw_real2, raw_gen2)

            grad_d = tape.gradient(discrim_loss_val, self.discriminator.trainable_variables)
            self.optimizer_d.apply_gradients(zip(grad_d, self.discriminator.trainable_variables))

            return p_gen_val, p_real_val, discrim_loss_val, gen_loss_val
                
        p_real = []
        p_fake = []
        discrim_loss = []
        gen_loss = []

        #Transform labels into OneHot-representation
        y_oh = OneHot(y, n=self.dim_y)

        for epoch in range(self.epochs):
            if (epoch + 1) % (0.1*self.epochs) == 0:
                print('Epoch:', epoch + 1)
            
            zs = np.random.normal(self.normal[0], self.normal[1], size=(len(y), self.dim_z))  
            
            ds_train = tf.data.Dataset.from_tensor_slices((x.astype(np.float32),
                                                           y_oh.astype(np.float32),
                                                           zs.astype(np.float32))).cache().shuffle(buffer_size=len(y))

            for xs, ys, zs in ds_train.batch(self.batch_size).prefetch(tf.data.AUTOTUNE):
                xs = tf.reshape(xs, [-1, 24, 24, 1])                  
                        
                if iterations % k == 0:
                    p_gen_val, p_real_val, discrim_loss_val, gen_loss_val = train_step_g(xs, ys, zs)                                  
                else:
                    p_gen_val, p_real_val, discrim_loss_val, gen_loss_val = train_step_d(xs, ys, zs)
                
                p_fake.append(p_gen_val)
                p_real.append(p_real_val)          
                discrim_loss.append(discrim_loss_val)
                gen_loss.append(gen_loss_val)

                if iterations % 1000 == 0:
                    print('Iterations',
                          iterations,
                          '| Average P(real):', f'{p_real_val:12.9f}',
                          '| Average P(fake):', f'{p_gen_val:12.9f}',
                          '| Discriminator Loss:', f'{discrim_loss_val:12.9f}',
                          '| Generator Loss:', f'{gen_loss_val:12.9f}')

                iterations += 1

        self.fitting_time = np.round(time.time() - self.fitting_time, 3)
        print('\nElapsed Training Time: ' + time.strftime('%Hh %Mmin %Ss', time.gmtime(self.fitting_time)))

        #Plotting
        fig, ax = plt.subplots()
        ax.plot(p_real, label='real')
        ax.plot(p_fake, label='fake')
        ax.legend()
        ax.set_xlim(0, len(p_real))
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Wasserstein Distance')
        ax.grid(True)
        fig.show()

        fig, ax = plt.subplots()
        ax.plot(discrim_loss)
        ax.set_xlim(0, len(discrim_loss))
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Discriminator Loss')
        ax.grid(True)
        fig.show()

    def predict(self):
        @tf.function
        def generate_step(zs, y_np_sample):
            return keras.ops.sigmoid(self.generator(zs, y_np_sample, training=False))
                
        generated_labels = np.random.randint(self.dim_y, size=(self.batch_size, 1))
        y_np_sample = OneHot(generated_labels, n=self.dim_y)
        zs = np.random.normal(self.normal[0], self.normal[1], size=(self.batch_size, self.dim_z))

        ds_generate = tf.data.Dataset.from_tensor_slices((zs.astype(np.float32),
                                                          y_np_sample.astype(np.float32))).cache()

        for zs, y_np_sample in ds_generate.batch(len(zs)).prefetch(tf.data.AUTOTUNE):
            generated_samples = generate_step(zs, y_np_sample)

        #Image shape 24x24 = 576
        generated_samples = np.reshape(generated_samples.numpy(), (-1, 576))

        return generated_samples, generated_labels
