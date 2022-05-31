# CV_assignments
2022_1 Computer Vision assignments
## 사용 방법
```shell
pip3 install -r requirements.txt
```
## 과제 내용
### 과제 1 : image filtering
이미지에 필터(커널)을 컨볼루션하여 필터링 하는 함수를 만듭니다.  
또한 만들어진 함수로 저주파 함수와 고주파 이미지를 합성하여 하이브리드 이미지를 생성합니다.  
![이미지](https://github.com/uichan8/CV_assignments/blob/main/hw1/results/hybrid_image_scales.jpg)  

### 과제 2 : feature matching
유사 Harris conner detection을 구현하여 특징점을 찾아내고 그 주변의 feature를 뽑는 함수를 구현합니다.  
이 벡터들을 비교하여 서로 다른 각도에서 찍은 두 사진의 정합점을 찾습니다.  
![이미지](https://github.com/uichan8/CV_assignments/blob/main/hw2/results/notre_dame_matches.jpg)  

### 과제 3 : bag of words
과제 2번이 특징점에서 벡터를 뽑았다면 이번에는 이미지 전체를 기준으로 벡터를 뽑습니다.
1. 이미지 벡터  
    tiny_image :  
     이미지를 16x16으로 나열하여 1 x 256 크기로 resize 한 벡터입니다.  
    bag_of_words :  
     학습할 이미지를 일정한 크기로 쪼갠 후 이 이미지들의 hog 벡터를 추출합니다.  
    각 이미지 벡터들을 k-means-cluster를 통해 대표적인 이미지 words를 선정합니다.   
    테스트할 이미지를 똑같은 크기로 쪼갠 후 hog 벡터를 추출합니다.  
    words 로 히스토그램을 만들어가장 유사한 항목에 포함시 킵니다. 이 히스토그램이 벡터가 됩니다.  

2. 분류기  
    knn :  
    테스트 벡터 로부터 가장 가까운 k개의 훈련용 벡터를 뽑습니다. k개의 벡터의 클래스중 가장 많은게 테스트 벡터의 클래스가 됩니다.  
    multi class linear svm :  
    벡터들의 클래스를 잘 나눠어주는 평면을 찾아 대입하여 클래스를 특정합니다.  

![이미지](https://github.com/uichan8/CV_assignments/blob/main/hw3/code/results_webpage/confusion_matrix.png)
      
### 과제 4 : DNN model(with pytorch)
