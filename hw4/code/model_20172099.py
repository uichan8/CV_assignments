import numpy as np
import matplotlib.pyplot as plt
import hyperparameters as hp
import random
from tqdm import tqdm

from sklearn.svm import LinearSVC

class Model:

    def __init__(self, train_images, train_labels, num_classes):
        self.input_size = train_images.shape[1] 
        self.num_classes = num_classes
        self.learning_rate = hp.learning_rate
        self.batchSz = hp.batch_size
        self.train_images = train_images
        self.train_labels = train_labels
        self.clf = LinearSVC(multi_class='ovr',dual=False)

        # sets up weights and biases...
        self.W = np.random.rand(self.input_size, self.num_classes)
        self.b = np.zeros((1, self.num_classes))

    def train_nn(self):
        #loss 를 그래프화 하기 위한 작업
        loss_array = []

        # 0부터 이미지갯수 만큼의 배열
        indices = list(range(self.train_images.shape[0])) 

        #한 퍼셉트론의 편미분 값
        delta_W = np.zeros((self.input_size, self.num_classes))
        delta_b = np.zeros((1, self.num_classes))

        #learning 시작
        for epoch in range(hp.num_epochs):
            # Overall per-epoch sum of the loss
            loss_sum = 0
            
            # 일정 애폭 지날시 학습율 
            if epoch == 40:
                self.learning_rate *= 0.1

            # 인덱스를 섞어줍니다.
            random.shuffle(indices)

            for index in range(len(indices)):
                # 이미지 가져오기
                i = indices[index]
                img = self.train_images[i]
                gt_label = self.train_labels[i]

                #feed foward
                linear = img @ self.W + self.b  #linear 
                soft_max = np.exp(linear-linear.max()) #activate
                soft_max /= np.sum(soft_max)

                #cross_entropy
                loss = -np.log2(np.maximum(soft_max[:,gt_label],0.00001))
                loss_sum += loss

                #back propergation
                #cross entropy 편미분
                delta_cross_entropy = np.zeros((1, self.num_classes))
                delta_cross_entropy[:,gt_label] = -1/np.log(2)/soft_max[:,gt_label] * loss 

                #soft max 편미분 #이렇게 한 이유는 cross entropy에서 class가 아닌경우도 log로 생각해줄 경우 작동하도록 만들음
                delta_softmax = -soft_max.T@soft_max
                delta_softmax += np.eye(self.num_classes)*soft_max
                delta_softmax *= delta_cross_entropy
                delta_softmax = delta_softmax.sum(axis=1).T

                #bias 편미분
                delta_b += delta_softmax

                #weight 편미분
                delta_W = (delta_W + delta_softmax) * img.reshape([-1,1])

                #W,b업데이트
                self.W -= delta_W * self.learning_rate
                self.b -= delta_b * self.learning_rate
                
            print( "Epoch " + str(epoch) + ": Total loss: " + str(loss_sum) )
            loss_array.append(loss_sum)
        plt.plot(np.arange(hp.num_epochs),loss_array)
        #plt.show()
        return loss_array


    def train_svm(self):
        """
        Use the response from the learned weights and biases on the training data
        as input into an SVM. I.E., train an SVM on the multi-class hyperplane distance outputs.
        """
        # Step 1:
        # Compute the response of the hyperplane on the training image
        scores = np.dot(self.train_images, self.W) + self.b

        # Step 2:
        # Fit an SVM model to these. Uses LinearSVC function declare in self.clf
        # This will be used later on in accuracy_svm()
        self.clf.fit(scores,self.train_labels)

    def accuracy_nn(self, test_images, test_labels):
        """
        Computer the accuracy of the neural network model over the test set.
        """
        scores = np.dot(test_images, self.W) + self.b
        predicted_classes = np.argmax(scores, axis=1)
        return np.mean(predicted_classes == test_labels)

    def accuracy_svm(self, test_images, test_labels):
        """
        Computer the accuracy of the svm model over the test set.
        """
        scores = np.dot(test_images, self.W) + self.b
        predicted_classes = self.clf.predict(scores)
        return np.mean(predicted_classes == test_labels)
