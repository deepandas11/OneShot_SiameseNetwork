import numpy as np
import os
from PIL import Image 
import random
import json
import keras

class DataGenerator(keras.utils.Sequence):

    def __init__(self, dim=(105,105), mode='train',
                 batch_size=32, shuffle=True,
                 support_size=10):

        self.dim = dim  # Dimension of images
        self.batch_size = batch_size  
        self.shuffle = shuffle 
        self.support_size = support_size
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

    def __len__(self):
        """
        Number of iterations in each epoch
        """
        return len(self.alphabets)


    def __getitem__(self,index):
        """
        Return full epoch's worth data as a list of batches
        If there are 24 alphabets in the fold, there will be 24 batches generated
        Thus, each epoch will have 24 steps
        """

        current_alphabet = self.alphabets[index]

        if self.mode in ['test','val']:
            X,y = self.get_eval_batch(alphabet=current_alphabet,
                                      support_size=self.support_size)
            return X,y

        X,y = self.get_train_batch(alphabet=current_alphabet)
        return X,y

    def on_epoch_end(self):
        """
        Shuffles order of alphabets to be fed in as batches
        """
        if self.shuffle == True:
            np.random.shuffle(self.alphabets)

    def get_train_batch(self, alphabet):
        """
        Get a batch of image pair paths and labels 
        The batch has n/2 pairs of images from the same class
        n/2 pairs of images that belong to different classes.
        """

        batch_paths = list()
        label_list = list()

        # Find the current alphabet
        current_alphabet = alphabet
        # Find all the characters in this alphabet 
        available_chars = list(self.image_paths[current_alphabet].keys())

        # For sampling with replacement. For cases where n_chars < batch_sz
        sz = self.batch_size//2
        batch_chars = list(np.random.choice(available_chars, size=sz))

        for character in batch_chars:
            # All images of the same character - all positive pairs. 
            anchor_images = self.image_paths[current_alphabet][character]

            # Sample 3 images from the character. 2 for +ve pair. 1 for -ve pair
            [ind1, ind2, ind3] = random.sample(range(0,20), 3)

            # Find a character from the remaining available characters
            other_characters = list(set(available_chars).difference(set([character])))
            character2 = random.sample(other_characters, k=1)[0]
            # Sample one image from the other character
            impostor_images = self.image_paths[current_alphabet][character2]
            [imp_ind] = random.sample(range(0,20), 1)

            current_path = os.path.join(self.image_folder, current_alphabet)
            # print(current_path, character, image_indices, impostor_image_index)

            # Paths to all image files
            image_file1 = os.path.join(current_path, character, anchor_images[ind1])
            image_file2 = os.path.join(current_path, character, anchor_images[ind2])
            image_file3 = os.path.join(current_path, character, anchor_images[ind3])
            impostor_image_file = os.path.join(current_path, character2, impostor_images[imp_ind])

            pos_sample = [image_file1, image_file2]
            neg_sample = [image_file3, impostor_image_file]

            batch_paths.append(pos_sample)
            batch_paths.append(neg_sample)
            label_list.append(1)
            label_list.append(0)

        image_pairs, labels = self.generate_data(batch_paths=batch_paths,
                                                 label_list=label_list)

        return image_pairs, labels

    def get_eval_batch(self, alphabet, support_size=10):
        """
        Generate a support_size sized batch of image pairs and labels
        This batch fixes an anchor character. Finds one similar character.
        Batch has 1 positive pair and rest are all negative pairs.
        """
        assert self.mode in ['val','test']

        batch_paths = list()
        label_list= list()

        current_alphabet = alphabet
        available_characters = list(self.image_paths[current_alphabet].keys())

        if support_size == -1 or (support_size > len(available_characters)):
            num_support_characters = len(available_characters)
        else:
            num_support_characters = support_size

        current_path = os.path.join(self.image_folder, current_alphabet)


        anchor_character = random.choice(available_characters)
        anchor_images = self.image_paths[current_alphabet][anchor_character]
        [ind1, ind2] = random.sample(range(0,20), 2)
        image_file1 = os.path.join(current_path, anchor_character, anchor_images[ind1])
        image_file2 = os.path.join(current_path, anchor_character, anchor_images[ind2])

        pos_pair = [image_file1, image_file2]
        label_list.append(1)
        batch_paths.append(pos_pair)

        other_characters = list(set(available_characters).difference(set(list(anchor_character))))
        impostor_characters = random.sample(other_characters, k=num_support_characters)

        for neg_character in impostor_characters:
            neg_images = self.image_paths[current_alphabet][neg_character]
            [imp_ind] = random.sample(range(0,20),1)
            image_file3 = os.path.join(current_path, neg_character, neg_images[imp_ind]) 

            neg_pair = [image_file1, image_file3]
            batch_paths.append(neg_pair)
            label_list.append(0)

        image_pairs, labels = self.generate_data(batch_paths, label_list, eval_flag=True)

        return image_pairs, labels


    def generate_data(self, batch_paths, label_list, eval_flag=False):
        """
        This function converts a list of paths to numpy image arrays
        Enables feeding to Keras Model.
        """
        number_of_pairs = len(batch_paths)  

        # Empty matrix of dimensions 2 x batch_size x image_size
        # Two batches of images 
        image_pairs = [np.zeros((number_of_pairs, *self.dim, 1)) for i in range(2)]
        labels = np.zeros((number_of_pairs, 1))

        for i in range(len(batch_paths)):
            pair = batch_paths[i]

            for j in range(len(pair)):  # j in range(2)
                image = Image.open(pair[j])
                image = np.asarray(image).astype(np.float64)
                image = image / image.std() - image.mean()
                image_pairs[j][i,:,:,0] = image

            labels[i] = label_list[i]

        if eval_flag:
            return image_pairs, labels

        random_permute = np.random.permutation(number_of_pairs)
        labels = labels[random_permute]
        for j in range(2):
            image_pairs[j][:,:,:,:] = image_pairs[j][random_permute,:,:,:]

        return image_pairs, labels