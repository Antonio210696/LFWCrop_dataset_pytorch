from torchvision.datasets import VisionDataset

from PIL import Image

import os
import string
import os.path
import sys
import torch

def pil_loader(image):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    img = Image.open(image)
    return img.convert('RGB')


class LFWCrop(VisionDataset):
    def __init__(self, root, transform=None):
        super(LFWCrop, self).__init__(
            root, transform=transform)


        '''
            This class has to retrieve the elements in LFWCrop dataset.
            Images in LFWCrop have the following naming schema:
            Name_Surname_#Image

            For the labels, we are going to get them from the list of attributes that are presenti in lfw_attributes.txt

            We are not going to use all the attributes that we have in the list, but just a subset.
        '''

        self.root = root
        self.transform = transform
        self.attributes_to_use = [
            "Male",
            "Asian",
            "White",
            "Black",
            "Baby",
            "Child",
            "Youth",
            "Middle Aged",
            "Senior",
            "Black Hair",
            "Blond Hair",
            "Brown Hair",
            "Bald",
            "No Eyewear",
            "Eyeglasses",
            "Sunglasses",
            "Mustache",
            "Smiling",
            "Frowning",
            "Chubby",
            "Blurry",
            "Harsh Lighting",
            "Flash",
            "Soft Lighting",
            "Outdoor",
            "Curly Hair",
            "Wavy Hair",
            "Straight Hair",
            "Receding Hairline",
            "Bangs",
            "Sideburns",
            "Fully Visible Forehead",
            "Partially Visible Forehead",
            "Obstructed Forehead",
            "Bushy Eyebrows",
            "Arched Eyebrows",
            "Narrow Eyes"
        #    "Eyes Open",
        #    "Big Nose",
        #    "Pointy Nose",
        #    "Big Lips",
        #    "Mouth Closed",
        #    "Mouth Slightly Open",
        #    "Mouth Wide Open",
        #    "Teeth Not Visible",
        #    "No Beard",
        #    "Goatee",
        #    "Round Jaw",
        #    "Double Chin",
        #    "Wearing Hat",
        #    "Oval Face",
        #    "Square Face",
        #    "Round Face",
        #    "Color Photo",
        #    "Posed Photo",
        #    "Attractive Man",
        #    "Attractive Woman",
        #    "Indian",
        #    "Gray Hair",
        #    "Bags Under Eyes",
        #    "Heavy Makeup",
        #    "Rosy Cheeks",
        #    "Shiny Skin",
        #    "Pale Skin",
        #    "5 o' Clock Shadow",
        #    "Strong Nose-Mouth Lines",
        #    "Wearing Lipstick",
        #    "Flushed Face",
        #    "High Cheekbones",
        #    "Brown Eyes",
        #    "Wearing Earrings",
        #    "Wearing Necktie",
        #    "Wearing Necklace"
        ]

        print("Using %d attributes", len(self.attributes_to_use))

        # Retrieving cropped actors names
        print("Root directory is " + self.root)
        self.cropped_actors_files = os.listdir(self.root + "faces")
        self.cropped_actors = []
        for actor in self.cropped_actors_files:
            actor = actor.rstrip('.ppm')
            self.cropped_actors.append(actor)
        print("Selected " + str(len(self.cropped_actors)) + " actors")

        # Retrieving selected attributes indeces
        self.att_indeces = []
        self.faces_dict = {}
        self.faces = []
        self.facesDir = 'faces/'
        att_file = open(self.root + "lfw_attributes.txt", 'r')

        for i, line in enumerate(att_file):
            if i == 1:
                line = line.split('\t')
                for att_name in line:
                    print("Selected attributes: " + att_name)
                    if att_name in self.attributes_to_use:
                        self.att_indeces.append(line.index(att_name))

            # Preparing images list
            elif i > 1:
                line = line.split('\t')
                actor_code = line[0].replace(' ', '_') + '_' + format(int(line[1]), '04')
                if actor_code in self.cropped_actors: 
                    imageFile = actor_code + '.ppm'
                    self.faces_dict[actor_code] = [float(line[index]) for index in self.att_indeces]

                    # Creating the final list of tuples(<pil_image>, <attributes_list>) 
                    image = open(self.root + self.facesDir + imageFile, 'rb')
                    image = pil_loader(image)
                    self.faces.append((image, self.faces_dict[actor_code]))




        # Taking the split files

        # First, we obtain the class_indeces for all classes

#        if self.split == 'train':
#            trainFile = open(root + "/train.txt", 'r')
#            self.pairs = []
#            # Discerning the label from the image file
#            for file in trainFile:
#                file = file.rstrip()
#                label = file.split('/')[0]
#                # if the file is the 'BACKGROUND_Google', we filter it out
#                if label != "BACKGROUND_Google":
#                    if full_load:
#                        # we open the entire file and put it in memory
#                        # It is suggested to do this only on the virtual environment of
#                        # COLab
#                        image = open(objCatDir + file, 'rb')
#                    else:
#                        image = objCatDir + file
#                    self.pairs.append((image, class_indeces[label]))
#                else:
#                    print(label)


        '''
        - Here you should implement the logic for reading the splits files and accessing elements
        - If the RAM size allows it, it is faster to store all data in memory
        - PyTorch Dataset classes use indexes to read elements
        - You should provide a way for the __getitem__ method to access the image-label pair
          through the index
        - Labels should start from 0, so for Caltech you will have lables 0...100 (excluding the background class) 
        '''

    def __getitem__(self, index):
        '''
        __getitem__ should access an element through its index
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        '''

        image, label = self.faces[index]
        label = torch.FloatTensor(label)
        #    image = pil_loader(image)

        #    fileName, label = self.pairs[index]
        #    image = open(fileName, 'rb')
        #    image = pil_loader(image)

        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        

        return image, label

    def __len__(self):
        '''
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        length  = len(self.cropped_actors)
        return length
