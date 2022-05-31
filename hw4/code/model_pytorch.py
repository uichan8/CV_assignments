import numpy as np
import random
import hyperparameters as hp

import torch
import torch.nn as nn
import torch.nn.functional as F
# import torchvision
# import torchvision.transforms as transforms


class Model_pytorch(nn.Module):
 
    def __init__(self, num_classes):
        super().__init__()
        """
        define the network layers for the convolutional neural network model.
        """




    def forward(self, x):
        """
        define the forward pass operations based on the defined network layers.
        """




        return 

def train_pytorch(train_images, train_labels, num_classes):
    """
    train the network by stochastic gradient descent.
    """
        
    # device options on whether to run the training on GPU or CPU.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # initialize model instance
    model = Model_pytorch(num_classes).to(device)
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
            # setup input and target label data mini-batch
            idxs = ind_list[i*hp.batch_size:i*hp.batch_size+hp.batch_size]
            img_batch = train_images[idxs,:]
            img_batch = img_batch.reshape(img_batch.shape[0], 1, hp.img_size, hp.img_size)
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