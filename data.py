import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
import PIL


class Data:
    def __init__(self, path):
        self.data_frame = pd.read_csv(path)
        self.data_frame['pixels'] = self.data_frame['pixels'].apply(lambda x: np.array(x.split(), dtype="float32"))
        # normalizing pixels data
        self.data_frame['pixels'] = self.data_frame['pixels'].apply(lambda x: x / 255)
        X = np.array(self.data_frame['pixels'].tolist())
        # Converting pixels from 1D to 3D
        self.images = X.reshape(X.shape[0], 48, 48, 1)
        self.genders = self.data_frame['gender']
        self.ages = self.data_frame['age']
        self.ethnicities = self.data_frame['ethnicity']

    def get_data(self, y, test_size):
        return train_test_split(
            self.images, y, test_size=test_size, random_state=37
        )



