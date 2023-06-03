from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import random
import numpy as np
from tqdm import tqdm


class DataManager:
    def __init__(self, size=-1):
        self.X = None
        self.Y = None
        self.training_set_size = None
        self.load_data(size)

    def load_data(self, size):         
        images = os.listdir("./celeba/")
        random.shuffle(images)
        if size!=-1:
            images = images[0:size]
        data = []
        res = []
        
        for i in tqdm(images):
            img =  cv2.cvtColor(cv2.imread("./celeba/"+i),cv2.COLOR_BGR2GRAY)
            data.append(cv2.resize(img,(32,32)))
            res.append(cv2.resize(img,(128,128)))

        data = np.array(data)
        data = data.reshape(data.shape[0], 32, 32, 1)
        res = np.array(res)
        res = res.reshape(res.shape[0], 128, 128, 1)
        
        self.X = data
        self.Y = res
        self.training_set_size = len(images)

    def get_batch(self, batch_size, index):  
        start = index*batch_size
        end = start+batch_size
        
        return self.Y[start:end],self.X[start:end]

    def shuffle(self):
        indices = np.arange(self.training_set_size)
        np.random.shuffle(indices)
        self.X = self.X[indices]
        self.Y = self.Y[indices]
