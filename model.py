import numpy as np
import tensorflow as tf
from keras import layers as L
from sklearn.model_selection import train_test_split
import cv2 as cv
from data import Data



class Model:
    ethnicity_classifications = ['white', 'black', 'asian', 'indian', 'other']
    gender_classifications = ['male', 'female']

    def __init__(self, input_shape=(48, 48, 1)):
        self.input_shape = input_shape
        self.data = Data("data/age_gender.csv")
        self.age_model = None
        self.gender_model = None
        self.ethnicity_model = None

    def build_model(self):
        print("+++++++++++++++++++++++++building age model+++++++++++++++++++++++++++++")
        self.age_model = self.get_age_model()
        print("+++++++++++++++++++++++++building gender model+++++++++++++++++++++++++++++")
        self.gender_model = self.get_gender_model()
        print("+++++++++++++++++++++++++building ethnicity model+++++++++++++++++++++++++++++")
        self.ethnicity_model = self.get_ethnicity_model()
        self.train()

    ## Stop training when validation loss reach 110
    class ageCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if (logs.get('val_loss') < 110):
                print("\nReached 110 val_loss so cancelling training!")
                self.model.stop_training = True

    ## Stop training when validation accuracy reach 79%
    class ethnicityCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if (logs.get('val_accuracy') > 0.790):
                print("\nReached 79% val_accuracy so cancelling training!")
                self.model.stop_training = True

    ## Stop training when validation loss reach 0.2700
    class genderCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if (logs.get('val_loss') < 0.2700):
                print("\nReached 0.2700 val_loss so cancelling training!")
                self.model.stop_training = True

    def get_age_model(self):
        model = tf.keras.Sequential([
            L.InputLayer(input_shape=(48, 48, 1)),
            L.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
            L.BatchNormalization(),
            L.MaxPooling2D((2, 2)),
            L.Conv2D(64, (3, 3), activation='relu'),
            L.MaxPooling2D((2, 2)),
            L.Conv2D(128, (3, 3), activation='relu'),
            L.MaxPooling2D((2, 2)),
            L.Flatten(),
            L.Dense(64, activation='relu'),
            L.Dropout(rate=0.5),
            L.Dense(1, activation='relu')
        ])
        return model

    def get_gender_model(self):
        model = tf.keras.Sequential([
            L.InputLayer(input_shape=(48, 48, 1)),
            L.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
            L.BatchNormalization(),
            L.MaxPooling2D((2, 2)),
            L.Conv2D(64, (3, 3), activation='relu'),
            L.MaxPooling2D((2, 2)),
            L.Flatten(),
            L.Dense(64, activation='relu'),
            L.Dropout(rate=0.5),
            L.Dense(1, activation='sigmoid')
        ])

        return model

    def get_ethnicity_model(self):
        model = tf.keras.Sequential([
            L.InputLayer(input_shape=(48, 48, 1)),
            L.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
            L.MaxPooling2D((2, 2)),
            L.Conv2D(64, (3, 3), activation='relu'),
            L.MaxPooling2D((2, 2)),
            L.Flatten(),
            L.Dense(64, activation='relu'),
            L.Dropout(rate=0.5),
            L.Dense(5)
        ])
        return model

    def train(self):
        self.ethnicity_model.compile(optimizer='rmsprop',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        self.compile_and_fit(self.ethnicity_model, self.data.ethnicities, self.ethnicityCallback(), "ethnicity")

        self.gender_model.compile(optimizer='sgd',
                      loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=['accuracy'])
        self.compile_and_fit(self.gender_model, self.data.genders, self.genderCallback(), "gender")

        sgd = tf.keras.optimizers.SGD(momentum=0.9)
        self.age_model.compile(optimizer='adam',
                      loss='mean_squared_error',
                      metrics=['mae'])
        self.compile_and_fit(self.age_model, self.data.ages, self.ageCallback(), "age")

    def compile_and_fit(self, model, y, callback, name, max_epochs=2000):
        X_train, X_test, y_train, y_test = train_test_split(self.data.images, y, test_size=0.22, random_state=37)
        model.summary()
        checkpoint_path = "drive/MyDrive/Colab Notebooks/{}/cp.ckpt".format(name)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         save_weights_only=True,
                                                         verbose=1)
        history = model.fit(X_train, y_train, epochs=100, validation_split=0.1, batch_size=64,
                            callbacks=[callback, cp_callback])
        eval(model, X_test, y_test)
        return history

    def predict(self, image):
        imported_image = cv.resize(image, (48, 48))
        image_pixels = np.expand_dims(np.asarray(imported_image), axis=0)
        image_pixels = image_pixels / 255
        age = self.age_model.predict(image_pixels)
        gender = self.gender_model.predict(image_pixels)
        ethnicity = self.ethnicity_model.predict(image_pixels)
        return f'Prediction: {age[0][0]} years old, {self.gender_classifications[int(gender[0][0])]}, {self.ethnicity_classifications[ethnicity[0].argmax()]}'
        # plt.title(
        #     f'Prediction: {age[0][0].argmax()} years, {self.gender_classifications[gender[0][0].argmax()]}, {self.ethnicity_classifications[ethnicity[0][0].argmax()]}\n')
        # plt.imshow(image_pixels)
        # plt.show()
