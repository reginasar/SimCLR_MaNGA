import tensorflow as tf


class BigCNN(tf.keras.Model):
    def __init__(self, out_dim):
        super(BigCNN, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(filters=128, kernel_size=5, padding='same', strides=1)#, kernel_initializer=initializer)
        self.conv2 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', strides=1)#, kernel_initializer=initializer)
        self.conv3 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='same', strides=1)#, kernel_initializer=initializer)
        self.conv4 = tf.keras.layers.Conv2D(filters=1024, kernel_size=3, padding='same', strides=1)#, kernel_initializer=initializer)
#        self.conv5 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', strides=1)#, kernel_initializer=initializer)

        self.l1 = tf.keras.layers.Dense(units=512)#, kernel_initializer=initializer)
        self.l2 = tf.keras.layers.Dense(units=128)#, kernel_initializer=initializer)
        self.l3 = tf.keras.layers.Dense(units=64)#, kernel_initializer=initializer)

        self.activation = tf.keras.layers.Activation('elu')
        self.max_pool = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)

        self.global_pool = tf.keras.layers.GlobalAveragePooling2D()

    def call(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.max_pool(x)

        x = self.conv2(x)
        x = self.activation(x)
        x = self.max_pool(x)

        x = self.conv3(x)
        x = self.activation(x)
        x = self.max_pool(x)

        x = self.conv4(x)
        x = self.activation(x)
        x = self.max_pool(x)

#        x = self.conv5(x)
#        x = self.activation(x)
#        x = self.max_pool(x)

        h = self.global_pool(x)

        x = self.l1(h)
        x = self.activation(x)
        x = self.l2(x)
        x = self.activation(x)
        x = self.l3(x)

        return h, x
