import tensorflow_datasets as tfds
import tensorflow as tf
import os
import cv2
import numpy as np


class Segmentator:

    def __init__(self):
        self.model = None
        self.batch_size = 28  # default value
        self.epochs = 1
        self.history = None
        self.callbacks = []

    def __preprocess(self, example):
        image = example['image']
        mask = example['mask']
        image = tf.image.convert_image_dtype(image, tf.float32)
        mask = tf.cast(mask, tf.float32)
        return image, mask

    def load_dataset(self, split):
        if not os.path.exists('dataset'):
            os.makedirs('dataset')
        dataset = tfds.load('my_custom_dataset', split=split, data_dir='dataset')
        dataset = dataset.map(self.__preprocess).batch(self.batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset

    def set_epochs(self,
                   epochs: int):
        """
        Sets the number of epochs to be used during model training the model. It checks whether the input
        value is a positive integer and assigns it to the `epochs` attribute of the class instance.
        """
        if isinstance(epochs, int) and epochs > 0:
            self.epochs = epochs

    def define_early_stopping(self, patience=8):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',       # Metric to monitor
                                                          patience=patience,    # Number of epochs
                                                          verbose=1,     # information about early_stopping
                                                          restore_best_weights=True  # Restore model weights
                                                          )
        self.callbacks.append(early_stopping)

    def define_reduce_lr(self, patience=4):
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                              factor=0.5,
                                                              patience=patience,
                                                              min_lr=1e-6,
                                                              verbose=1,
                                                              mode='min')
        self.callbacks.append(reduce_lr)

    def postprocess_image(selfs, image):
        prd_mask = np.squeeze(image, axis=0)
        mask = tf.argmax(prd_mask, axis=-1)
        return mask

    def preprocess_image(self, image):

        resized_image = cv2.resize(image, (240, 320))
        normalized_image = resized_image.astype(np.float32) / 255.0
        image = np.expand_dims(normalized_image, axis=0)

        return image

    def set_batch_size(self,
                       batch_size: int):
        """
        Sets the batch size for training and evaluation of the model.

        Args:
            batch_size (int): The batch size for training and evaluation.
        """
        if isinstance(batch_size, int) and batch_size > 0:
            self.batch_size = batch_size

    def build_model(self, input_size=(320, 240, 3)):

        inputs = tf.keras.layers.Input(input_size)

        # Encoder
        conv1 = tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same')(inputs)
        pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(pool1)
        pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

        # Bottleneck
        conv3 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(pool2)

        # Decoder
        up4 = tf.keras.layers.Conv2D(32, 2, activation='relu', padding='same')(
            tf.keras.layers.UpSampling2D(size=(2, 2))(conv3))
        merge4 = tf.keras.layers.concatenate([conv2, up4], axis=3)
        conv4 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(merge4)

        up5 = tf.keras.layers.Conv2D(16, 2, activation='relu', padding='same')(
            tf.keras.layers.UpSampling2D(size=(2, 2))(conv4))
        merge5 = tf.keras.layers.concatenate([conv1, up5], axis=3)
        conv5 = tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same')(merge5)

        conv6 = tf.keras.layers.Conv2D(11, 1, activation='softmax')(conv5)

        model = tf.keras.models.Model(inputs=inputs, outputs=conv6)
        # mean_iou = tf.keras.metrics.MeanIoU(num_classes=11)
        # Precision calculation

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        self.model = model

    def build_model_2(self, input_size=(320, 240, 3)):

        inputs = tf.keras.layers.Input(input_size)

        c1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        c1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
        p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

        c2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
        c2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c2)
        p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

        c3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
        c3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c3)
        p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

        # Bottleneck
        c4 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p3)
        c4 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c4)

        # Expansive path
        u5 = tf.keras.layers.UpSampling2D((2, 2))(c4)
        u5 = tf.keras.layers.concatenate([u5, c3])
        c5 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u5)
        c5 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c5)

        u6 = tf.keras.layers.UpSampling2D((2, 2))(c5)
        u6 = tf.keras.layers.concatenate([u6, c2])
        c6 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u6)
        c6 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c6)

        u7 = tf.keras.layers.UpSampling2D((2, 2))(c6)
        u7 = tf.keras.layers.concatenate([u7, c1])
        c7 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u7)
        c7 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c7)

        outputs = tf.keras.layers.Conv2D(1176, (1, 1), activation='softmax')(c7)

        model = tf.keras.models.Model(inputs=[inputs], outputs=[outputs])

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                         loss='sparse_categorical_crossentropy',
                         metrics=[tf.keras.metrics.MeanIoU(num_classes=11, sparse_y_pred=False)])

        self.model = model

    def save_model(self, path_to_save):
        """
        This function saves the current state of the model, including its architecture, weights, and configuration,
        to a specified file path.

        Args:
            path_to_save (str): The file path where the model should be saved (filepath or directory).
        """
        self.model.save(path_to_save)

    def train_model(self, train_dataset, val_dataset):
        """
        Trains the CNN model on the provided training dataset and validates it on the validation dataset.
        It also supports early stopping through the use of callbacks to prevent overfitting.
        Notes:
            - `self.epochs` the number of epochs to train the model.
            - `self.callbacks` defined within the class and may include an EarlyStopping callback.
            - `self.model` a pre-built and compiled Keras Model object.
            - Training history is stored in `self.history` for further analysis.
        """
        self.model.fit(
            train_dataset,
            epochs=self.epochs,
            validation_data=val_dataset,
            callbacks=self.callbacks  # Include EarlyStopping callback
        )

    def eval_set(self, dataset):
        test_loss, test_iou = self.model.evaluate(dataset)
        return test_loss, test_iou
