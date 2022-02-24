import os
import numpy as np
from imgaug import augmenters as iaa 
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import math

# Basic Augment Object 
# image_paths: "path of the image"
# labels: "various label for image"
# show_samples: "True or False"
class BasicAugment():
    def __init__(self, image_paths, labels, show_sample=False, scale=1.3, translate_percent={"x":(-0.1, 0.1), "y": (-0.1, 0.1)}, brightness=(0.2, 1.2)):
        self.image_paths = image_paths
        self.labels = labels
        self.show_sample = show_sample
        
        self.scale = scale
        self.translate_percent = translate_percent
        self.brightness = brightness

        self.show_sample_image()
        
    def batch_generator(self, batch_size, istraining):
        while True:
            batch_img = []
            batch_label = []
            
            for _ in range(batch_size):
                random_index = random.randint(0, len(self.image_paths)-1)    
                label = self.labels[random_index]
                
                if istraining:
                    im = self.random_augment(self.image_paths[random_index])        
                    
                else:
                    im = mpimg.imread(self.image_paths[random_index])
            
                im = self.preprocess(im)
                batch_img.append(im)
                batch_label.append(label)
            yield (np.asarray(batch_img), np.asarray(batch_label))
    
    def generate(self, size):
        new_img = []
        new_label = []
        for _ in range(size):
            random_index = random.randint(0, len(self.image_paths)-1)
            label = self.labels[random_index]
            
            im = self.random_augment(self.image_paths[random_index])
            
            im = self.preprocess(im)
            new_img.append(im)
            new_label.append(label)
            
        return np.asarray(new_img), np.asarray(new_label)


    def preprocess(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        img = cv2.GaussianBlur(img,  (3, 3), 0)
        img = cv2.resize(img, (200, 66))
        img = img/255
        return img  

        
    def random_augment(self, image):
        image = mpimg.imread(image)
        if np.random.rand() < 0.5:
            image = self.pan(image, self.translate_percent)
        if np.random.rand() < 0.5:
            image = self.zoom(image, self.scale)
        if np.random.rand() < 0.5:
            image = self.img_random_brightness(image, self.brightness)
        if np.random.rand() < 0.5:
            image = self.img_random_flip(image)
        
        return image
        
    def show_sample_image(self):
        if self.show_sample:
            random_index = random.randint(0, len(self.image_paths)-1)
            
            original_image = mpimg.imread(self.image_paths[random_index])
            augmented_image = self.random_augment(self.image_paths[random_index])
            
            self.show_sample_image_plot(original_image, augmented_image)


    
    def show_sample_image_plot(self, original_image, augmented_image):    
        fig, axs = plt.subplots(1, 2, figsize=(15, 10))
        fig.tight_layout()
            
        axs[0].imshow(original_image)
        axs[0].set_title("Original Image")
            
        axs[1].imshow(augmented_image)
        axs[1].set_title("Augmented Image")
            
        plt.show()
    
    @staticmethod
    def zoom(image, scale):
        zoom = iaa.Affine(scale=(1, scale))
        image = zoom.augment_image(image)
        
        return image 

    @staticmethod
    def pan(image, translate_percent):
        pan = iaa.Affine(translate_percent=translate_percent)
        image = pan.augment_image(image)
        
        return image
    
    @staticmethod
    def img_random_brightness(image, brightness):
        brightness = iaa.Multiply(brightness)
        image = brightness.augment_image(image)
        
        return image
    
    @staticmethod
    def img_random_flip(image):
        image = cv2.flip(image, 1)    
        
        return image
    
    
if __name__ == '__main__':
    images = ["cat1.jpeg", "cat2.jpeg", "dog1.jpeg", "dog2.jpeg"]
    labels = ["cat", "cat", "dog", "dog"]
    image_paths = []

    for image in images:
        image_paths.append(os.path.join("dataset", image))
    
    print(image_paths)    
        
    augment_generator = BasicAugment(image_paths, labels, False)

    size = 10
    X, Y = augment_generator.generate(size)
    
    ncol = 2
    nrow = math.ceil(size / ncol)  

    fig, axs = plt.subplots(nrow, ncol, figsize=(15, 50))
    fig.tight_layout()
        
    for i in range(len(X)):
        
        # augmented_image = mpimg.imread(X[i])
        
        axs[math.ceil(i/2)-1][i%2].imshow(X[i])
        axs[math.ceil(i/2)-1][i%2].set_title(f"Augmented Image {Y[i]}")
    
    plt.show()