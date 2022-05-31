import numpy as np
import random
import hyperparameters as hp

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from skimage import io
from skimage.transform import resize
import matplotlib.pyplot as plt
# from skimage.transform import resize
# import torchvision
# import torchvision.transforms as transforms


class Model_pytorch(nn.Module):
 
    def __init__(self, num_classes,input_size):
        super().__init__() 
        fc_layer_size = int((input_size - 4)/2/2)
        self.conv1 = nn.Conv2d(1,64,5) 

        self.shortcut2 = nn.Sequential()
        self.conv2 = nn.Sequential(
            nn.Conv2d(64,64,3, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,64,3, padding = 1),
            nn.BatchNorm2d(64)
        )
        self.ReLU = nn.ReLU()
        
        self.pool2 = nn.Sequential( 
            nn.ReLU(),
            nn.MaxPool2d(2)
        )


        self.full = nn.Sequential(
            nn.Flatten(),
            nn.Linear((fc_layer_size**2)*64, 120),
            nn.Linear(120, 84),
            nn.Linear(84, num_classes)
        )

        # self.shortcut3 = nn.Sequential()
        # self.conv3 = nn.Sequential(
        #     nn.Conv2d(64,128,3, padding = 1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        #     nn.Conv2d(128,128,3, padding = 1),
        #     nn.BatchNorm2d(128)
        # )
        # self.pool3 = nn.Sequential(
        #     nn.ReLU(),
        #     nn.MaxPool2d(2) #15
        # )

    def forward(self, x): 
        x = self.conv1(x) 

        sc = self.shortcut2(x)
        x = self.conv2(x)
        x += sc
        x = self.ReLU(x) 

        sc = self.shortcut2(x)
        x = self.conv2(x)
        x += sc
        x = self.pool2(x) 

        sc = self.shortcut2(x)
        x = self.conv2(x)
        x += sc
        x = self.ReLU(x) 

        sc = self.shortcut2(x)
        x = self.conv2(x)
        x += sc
        x = self.pool2(x) 

        sc = self.shortcut2(x)
        x = self.conv2(x)
        x += sc
        x = self.ReLU(x) 

        out = self.full(x)
        #out = nn.Softmax(x,dim = 1)
        return out
def load_data_scene(search_path, categories, size):
    images = np.zeros((size * hp.scene_class_count, hp.img_size * hp.img_size))
    labels = np.zeros((size * hp.scene_class_count,), dtype = np.int8)
    for label_no in range(hp.scene_class_count):
        img_path = search_path + categories[label_no]
        img_names = [f for f in os.listdir(img_path) if ".jpg" in f]
        for i in range(size):
            im = io.imread(img_path + "/" + img_names[i])
            im_vector = resize(im, (hp.img_size, hp.img_size)).reshape(1, hp.img_size * hp.img_size)
            index = size * label_no + i
            images[index, :] = im_vector
            labels[index] = label_no
    return images, labels

def get_categories_scene(search_path):
    dir_list = []
    for filename in os.listdir(search_path):
        if os.path.isdir(os.path.join(search_path, filename)):
            dir_list.append(filename)
    return dir_list

def format_data_scene_rec():
    train_path = "../data/train/"
    test_path = "../data/test/"
    categories = get_categories_scene(train_path)    
    train_images, train_labels = load_data_scene(train_path, categories, hp.num_train_per_category)
    test_images, test_labels = load_data_scene(test_path, categories, hp.num_test_per_category)
    return train_images, train_labels, test_images, test_labels

def train_pytorch(train_images, train_labels, num_classes):
    print("resnet")
    """
    train the network by stochastic gradient descent.
    """
    train_images, train_labels, test_images, test_labels = format_data_scene_rec()
    num_classes = hp.scene_class_count
        
    # device options on whether to run the training on GPU or CPU.
    print(f"cuda : {torch.cuda.is_available()}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    accuracy_array = []

    # initialize model instance
    image_size = int(np.sqrt(train_images[1,:].shape))
    model = Model_pytorch(num_classes,image_size).to(device)
    model.train()

    # setting the optimizer with the model parameters and learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=hp.learning_rate, momentum=hp.momentum)

    # setting the loss function
    cost = nn.CrossEntropyLoss()

    # number of training images
    num_trimg = train_images.shape[0]
    # set number of main training iterations
    total_step = num_trimg//hp.batch_size
    # change 0.1 to control number of print outputs during one epoch
    print_interval = int(total_step*0.1)
    
    # iterative SGD loop
    for epoch in range(hp.num_epochs):

        # to shuffle data for each epoch
        ind_list = list(range(num_trimg))
        random.shuffle(ind_list)
        
        for i in range(total_step):
            #if total_step == 50:
            #    optimizer = torch.optim.SGD(model.parameters(), lr=hp.learning_rate*0.5, momentum=hp.momentum)
            # setup input and target label data mini
            idxs = ind_list[i*hp.batch_size:i*hp.batch_size+hp.batch_size]
            img_batch = train_images[idxs,:]
            #img_batch = img_batch.reshape(img_batch.shape[0], 1, hp.img_size, hp.img_size)
            img_batch = img_batch.reshape(img_batch.shape[0], 1, image_size, image_size)
            images = torch.from_numpy(img_batch).float()
            images = images.to(device)
            lbl_batch = train_labels[idxs]
            labels = torch.from_numpy(lbl_batch).long()
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = cost(outputs, labels)
                
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                    
            # Print batch loss at print_intervals
            if (i+1) % print_interval == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                            .format(epoch+1, hp.num_epochs, i+1, total_step, loss.item()))
        accuracy = test_pytorch(model, test_images, test_labels)
        accuracy_array.append(accuracy)
        print(f"accuracy : {accuracy*100:.2f}%")
    plt.plot(np.arange(len(accuracy_array)),accuracy_array)
    plt.show()
            

    # return trained model
    return model

def test_pytorch(model, test_images, test_labels):

    # Device will determine whether to run the training on GPU or CPU.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    # for all test images, do inference and count corrent ones
    num_test_imgs = test_images.shape[0]
    num_correct = 0
    with torch.no_grad():
        for i in range(num_test_imgs):

            # setup i_th test image
            img= test_images[i,:]
            img = img.reshape(1, 1, hp.img_size, hp.img_size)
            image = torch.from_numpy(img).float()
            image = image.to(device)

            # perform inference
            outputs = model(image)
            # outputs = torch.exp(outputs-torch.max(outputs))
            # outputs = outputs/outputs.sum()
            _, predicted_class = torch.max(outputs, axis=1)

            # count correct outcomes
            if (predicted_class == test_labels[i]):
                num_correct += 1

    # return accuracy as ratio of correctly inferred images
    return num_correct/num_test_imgs