import numpy as np
import os
from PIL import Image 
import random
import json

class DataGenerator():

    def __init__(self, dim=(105,105), mode='train',batch_size=32, shuffle=True):
        self.dim = dim  # Dimension of images
        self.batch_size = batch_size  
        self.shuffle = shuffle 
        self.mode = mode  # Mode of operation 
        if self.mode in ['train','val']:
            self.image_folder = os.path.join('data','images_background')
            if self.mode == 'train':              
                self.data_file = os.path.join('data','train_data.json')
                with open(self.data_file, mode='r', encoding='utf-8') as f:
                    self.image_paths = json.load(f)  # Data dictionary
                self.alphabets = list(self.image_paths.keys())  # All the alphabets in this fold
                self.num_alphabets = len(self.alphabets)  # Number of alphabets in this fold
                self.current_alphabet_index = 0  # Counter variable 
            else:
                self.data_file = os.path.join('data', 'val_data.json')
                with open(self.data_file, mode='r', encoding='utf-8') as f:
                    self.image_paths = json.load(f)
                self.alphabets = list(self.image_paths.keys())
                self.num_alphabets = len(self.alphabets)
                self.current_alphabet_index = 0

        else:
            self.image_folder = os.path.join('data', 'images_evaluation')
            self.data_file = os.path.join('data','eval_data.json')
            with open(self.data_file, mode='r', encoding='utf-8') as f:
                self.image_paths = json.load(f)
            self.alphabets = list(self.image_paths.keys())
            self.num_alphabets = len(self.alphabets)
            self.current_alphabet_index = 0


    def get_train_batch(self):

        batch_paths = list()
        label_list = list()

        # Find the current alphabet
        current_alphabet = self.alphabets[self.current_alphabet_index]
        # Find all the characters in this alphabet 
        available_chars = list(self.image_paths[current_alphabet].keys())

        # For sampling with replacement. For cases where n_chars < batch_sz
        sz = self.batch_size//2
        batch_chars = list(np.random.choice(available_chars, size=sz))

        for character in batch_chars:
            # All images of the same character - all positive pairs. 
            anchor_images = self.image_paths[current_alphabet][character]

            # Sample 3 images from the character. 2 for +ve pair. 1 for -ve pair
            image_indices = random.sample(range(0,20), 3)

            # Find a character from the remaining available characters
            other_characters = list(set(available_chars).difference(set([character])))
            character2 = random.sample(other_characters, k=1)[0]
            # Sample one image from the other character
            impostor_images = self.image_paths[current_alphabet][character2]
            impostor_image_index = random.sample(range(0,20), 1)

            current_path = os.path.join(self.image_folder, current_alphabet)
            # print(current_path, character, image_indices, impostor_image_index)

            # Paths to all image files
            image_file1 = os.path.join(current_path, character, anchor_images[image_indices[0]])
            image_file2 = os.path.join(current_path, character, anchor_images[image_indices[1]])
            image_file3 = os.path.join(current_path, character, anchor_images[image_indices[2]])
            impostor_image_file = os.path.join(current_path, character2, impostor_images[impostor_image_index[0]])

            pos_sample = [image_file1, image_file2]
            neg_sample = [image_file3, impostor_image_file]

            batch_paths.append(pos_sample)
            batch_paths.append(neg_sample)
            label_list.append(0)
            label_list.append(1)

        self.current_alphabet_index += 1
        if self.current_alphabet_index >= self.num_alphabets:
            self.current_alphabet_index = 0
        image_pairs, labels = self.gen_train_data_from_paths(batch_paths=batch_paths, 
                                                             label_list=label_list)

        return image_pairs, labels

    def gen_train_data_from_paths(self, batch_paths, label_list):

        number_of_pairs = self.batch_size  

        # Empty matrix of dimensions 2 x batch_size x image_size
        # Two batches of images 
        image_pairs = [np.zeros((number_of_pairs, *self.dim)) for i in range(2)]
        labels = np.zeros((number_of_pairs, 1))

        for i in range(len(batch_paths)):
            pair = batch_paths[i]

            for j in range(len(pair)):  # j in range(2)
                image = Image.open(pair[j])
                image = np.asarray(image).astype(np.float64)
                image = image / image.std() - image.mean()
                image_pairs[j][i,:,:] = image

            labels[i] = label_list[i]

        random_permute = np.random.permutation(number_of_pairs)
        labels = labels[random_permute]
        for j in range(2):
            image_pairs[j][:,:,:] = image_pairs[j][random_permute,:,:]

        return image_pairs, labels






