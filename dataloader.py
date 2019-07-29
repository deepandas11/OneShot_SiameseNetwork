import numpy as np
import keras
import os
from PIL import Image 


class DataGenerator(keras.utils.Sequence):

    def __init__(self, dim, mode='train',batch_size=32, shuffle=True):
        self.dim = (105,105)
        self.batch_size = batch_size
        self.shuffle = shuffle 
        self.mode = mode
        if self.mode = 'train':
            self.image_path = os.path.join('data','images_background')
            self.data_file = os.path.join('data','train_data.json')
            with open(self.data_file, mode='r', encoding='utf-8') as f:
                self.label_data = json.load(f)
            self.image_ids = list(self.label_data.keys())

        else:
            self.image_path = os.path.join('data','images_evaluation')
            self.data_file = os.path.join('data', 'test_data.json')
            with open(self.data_file, mode='r', encoding='utf-8') as f:
                self.label_data = json.load(f)
            self.image_ids = list(self.label_data.keys()) # Image paths

        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""

        return int(np.floor(len(self.image_ids)/ self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""

        # Find indices of a batch size and the image ids/paths
        indices = self.indexes[index*self.batch_size : (index+1)*self.batch_size]
        list_image_ids = [self.image_ids[index] for index in indices]

        X,y = self.__data_generation(list_image_ids)

        return X,y 

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.image_ids))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_image_ids):
        """
        Generates data containing batch_size samples with labels
        """

        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size), dtype = int)

        for i, ID in enumerate(list_image_ids):
            image_path = os.path.join(self.image_path, ID)
            im = Image.open(image_path)
            x[i,] = np.array(im)
            y[i] = self.label_data[ID][0]

        return X, y
