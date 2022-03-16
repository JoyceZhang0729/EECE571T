import PIL
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as L
from matplotlib import pyplot as plt

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
        self.build_model()

    def build_model(self):
        self.age_model = self.get_age_model()
        self.gender_model = self.get_gender_model()
        self.ethnicity_model = self.get_ethnicity_model()

    def get_age_model(self):
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
        self.compile_and_fit(model, self.data.ages)

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
        self.compile_and_fit(model, self.data.genders)
        return model

    def get_ethnicity_model(self):
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
        self.compile_and_fit(model, self.data.ethnicities)
        return model

    def compile_and_fit(self, model, y, max_epochs=10000):
        X_train, X_test, y_train, y_test = self.data.get_data(y, test_size=0.22)
        model.compile(optimizer='sgd',
                      loss='mean_squared_error',
                      metrics=[
                          'mae',
                          'accuracy'])
        model.summary()
        history = model.fit(
            X_train,
            y_train,
            epochs=max_epochs,
            callbacks=[self.myCallback()],
            verbose=0)

        self.eval(model, X_test, y_test)
        return history

    def eval(self, model, X_test, y_test):
        mse, mae = model.evaluate(X_test, y_test, verbose=0)
        print('Test Mean squared error: {}'.format(mse))
        print('Test Mean absolute error: {}'.format(mae))

    def predict(self, image_path):
        imported_image = PIL.Image.open(image_path).resize((48, 48)).convert('L')
        image_pixels = np.expand_dims(np.asarray(imported_image), axis=0)
        age = self.age_model.predict(image_pixels)
        gender = self.gender_model.predict(image_pixels)
        ethnicity = self.ethnicity_model.predict(image_pixels)
        plt.title(
            f'Prediction: {age[0][0].argmax()} years, {self.gender_classifications[gender[0][0].argmax()]}, {self.ethnicity_classifications[ethnicity[0][0].argmax()]}\n')
        plt.imshow(image_pixels)
        plt.show()

    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if (logs.get('val_loss') < 110):
                print("\nReached 110 val_loss so cancelling training!")
                self.model.stop_training = True

