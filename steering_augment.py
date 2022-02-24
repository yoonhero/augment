import matplotlib.image as mpimg
import numpy as np
from basic_augment import BasicAugment
import random

class SteeringAugment(BasicAugment):
    def batch_generator(self, batch_size, istraining):
        while True: 
            batch_img = []
            batch_label = []
            
            for _ in range(batch_size):
                random_index = random.randint(0, len(self.image_paths)-1)
                label = self.labels[random_index]
                
                if istraining:
                    im = self.random_augment(self.image_paths[random_index], self.labels[random_index])

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
            random_index = random.randint(0, len(self.image_path)-1) 
            label = self.labels[random_index]
            
            im = self.random_augment(self.image_paths[random_index], self.labels[random_index])               

            im = self.preprocess(im)
            new_img.append(im)
            new_label.append(label)

        return np.asarray(new_img), np.asarray(new_label)

    def random_augment(self, image, steering_angle):
        image = mpimg.imread(image)

        if np.random.rand() < 0.5:
            image = self.pan(image, self.translate_percent)
        if np.random.rand() < 0.5:
            image = self.zoom(image, self.scale)
        if np.random.rand() < 0.5:
            image = self.img_random_brightness(image, self.brightness)
        if np.random.rand() < 0.5:
            image, steering_angle = self.img_random_flip(image, steering_angle)
        
        return image, steering_angle
    
    def preprocess(self, img):
        img = img[60:135,:,:]
        img = super().preprocess(img)
        
        return img
    
    def show_sample_image(self):
        if self.show_sample:
            random_index = random.randint(0, len(self.image_paths)-1)
            
            original_image = mpimg.imread(self.image_paths[random_index])
            augmented_image = self.random_augment(self.image_paths[random_index], self.labels[random_index])
            
            self.show_sample_image_plot(original_image, augmented_image)

    @staticmethod
    def img_random_flip(image, steering_angle):
        image = super().img_random_flip(image)
        steering_angle = -steering_angle

        return image, steering_angle
        
        
        