from ast import Num
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, feature, img_as_int
from skimage.measure import regionprops
from skimage.transform import rescale,rotate

def show_image(image):
    """min_max scaling을 통해 범위를 0~255사이로 바꿔 이미지를 보여줍니다."""
    tem = (image - image.min()) / (image.max() - image.min()) * 255
    plt.imshow(tem,cmap="gray")
    plt.show()

def crop_image(image,center:list,window_size):
    """
    이미지를 중심을 기준으로 윈도우를 크롭합니다.
        input:
            center : list [y,x]
    """
    #call by reference, return reference로 바꿀 수 있는지 확인하기
    from_center = window_size//2
    return image[center[0]-from_center:center[0]+window_size-from_center,center[1]-from_center:center[1]+window_size-from_center]

def image_diffx(image):
    """이미지 x방향 편미분"""
    Lx = np.zeros(image.shape)
    Lx[:,:-1] += (image[:,1:])
    Lx[:,1:] -= (image[:,:-1])
    #Lx = filters.sobel_v(image)
    return Lx

def image_diffy(image):
    """이미지 y방향 편미분"""
    Ly = np.zeros(image.shape)
    Ly[:-1,:] += (image[1:,:])
    Ly[1:,:] -= (image[:-1,:])
    #Ly = filters.sobel_h(image)
    return Ly

def gaussian_filter(sigma,size):
    """짝수 커널용 가우시안 필터(과제 1 에서 나온 함수를 변형)"""
    s = sigma
    probs = np.asarray([np.exp(-(z+0.5)**2/(2*s*s))/np.sqrt(2*np.pi*s*s) for z in range(-size//2,size//2)], dtype=np.float32)
    kernel = np.outer(probs, probs)
    kernel = kernel / kernel.sum()
    return kernel

def make_histogram(magnitude,orientation,histogram_size) -> list:
    """정사각형 크기와 방향을 받아서 n개의 방향히스토그램으로 만들어 리턴"""
    magnitude = magnitude.reshape([-1])
    orientation = orientation.reshape([-1])
    orientation = np.array(orientation//(2*np.pi/histogram_size),dtype=int)
    histogram = np.zeros(histogram_size)
    for i in range(len(magnitude)):
        if orientation[i] >= histogram_size:
            orientation[i] -= 8
        histogram[orientation[i]] += magnitude[i]
    return list(histogram)

def get_interest_points(image, feature_width):
    #이미지 가우시안 필터링
    image = filters.gaussian(image,1.2)

    #이미지 미분
    IxIx = filters.gaussian(np.gradient(np.gradient(image, axis = 1),axis = 1),1)
    IyIy = filters.gaussian(np.gradient(np.gradient(image, axis = 0),axis = 0),1)
    IxIy = filters.gaussian(np.gradient(np.gradient(image, axis = 1),axis = 0),1)

    #C의 계산 및 thresholding
    alpha = 0.06
    det = IxIx*IyIy - IxIy ** 2
    trace = IxIx + IyIy
    cornerness_score = det - alpha * (trace**2)

    threshold = 0.0001
    cornerness_score = cornerness_score*(cornerness_score>=threshold)

    # 특징점 선별
    local_max = feature.peak_local_max(cornerness_score,min_distance=10)

    #adaptive Non-maximal suppression
    interest_arglist = []
    for i in range(local_max.shape[0]):
        edge = 20
        if edge < local_max[i,0] and local_max[i,0] < image.shape[0] - edge and edge < local_max[i,1] and local_max[i,1] < image.shape[1] - edge: # 가장자리에 있는 점들을 지우는 것
            window_size = 7 # 이 크기가 peak_local_max에 min_distance와 비슷한 역할을 합니다.
            crop_cornerness = crop_image(cornerness_score,local_max[i],window_size)
            max = crop_cornerness[window_size//2,window_size//2]
            crop_cornerness[window_size//2,window_size//2] = 0
            if max > 1.1 * crop_cornerness.max(): #주변의 픽셀보다 10% 더 큰지 확인하는 과정입니다.
                interest_arglist.append(i)

    local_max = local_max[interest_arglist]

    #구해진 값을 리턴합니다.
    xs = local_max[:,1]
    ys = local_max[:,0]

    return xs, ys


def get_interest_points2(image, feature_width):
    """Harris 88"""
    #이미지 가우시안 필터링
    image = filters.gaussian(image,1.2)

    # 이미지 1계도 미분
    window_sigma = 0
    Ix = filters.gaussian(image_diffx(image),window_sigma)
    Iy = filters.gaussian(image_diffy(image),window_sigma)

    # 모멘텀 행렬들의 원소를 구해줍니다., window function도 적용해줍니다. -> 대칭행렬
    window_sigma = 2
    IxIx = filters.gaussian(Ix**1,window_sigma)
    IxIy = filters.gaussian(Ix*Iy,window_sigma)
    IyIy = filters.gaussian(Iy**1,window_sigma)

    #cornerness score을 구합니다 -> 고유값이 동시에 큰 즉 어느 방향 방향으로 이동해도 변화가 큰 값을 잡을 수 있도록 수치화 해줍니다.
    alpha = 0.06
    determinant = IxIx*IyIy - IxIy**2
    trace = IxIx + IyIy
    cornerness_score = determinant - alpha * (trace ** 2)

    #adaptive Non-maximal suppression 을 적용하여 점을 선별해줍니다. 모서리의 값들을 버려줍니다. (특징점이 되기 위해서는 max일 뿐만아니라 주변 픽셀보다 값이 커야한다는 조건입니다.)
    threshold = 0.0001
    cornerness_score[cornerness_score < threshold] = 0
    local_max = feature.peak_local_max(cornerness_score, min_distance = 10)

    interest_arglist = []
    for i in range(local_max.shape[0]):
        edge = 20
        if edge < local_max[i,0] and local_max[i,0] < image.shape[0] - edge and edge < local_max[i,1] and local_max[i,1] < image.shape[1] - edge: # 가장자리에 있는 점들을 지우는 것
            window_size = 5
            crop_cornerness = crop_image(cornerness_score,local_max[i],window_size)
            max = crop_cornerness[window_size//2,window_size//2]
            crop_cornerness[window_size//2,window_size//2] = 0
            if max > 1.1 * crop_cornerness.max(): #주변의 픽셀보다 10% 더 큰지 확인하는 과정입니다.
                interest_arglist.append(i)

    local_max = local_max[interest_arglist]

    #구해진 값을 리턴합니다.
    xs = local_max[:,1]
    ys = local_max[:,0]
    return xs, ys

def get_features(image, x, y, feature_width):
    """SIFT 기술자 추출"""
    #이미지에 가우시안 필터를 적용합니다.
    image = filters.gaussian(image,2)

    #이미지 크기를 변경하고, 좌표를 반올림하여 정수를 만듭니다.
    window_image_size = 32 # 16, 32, 64 ...
    scale = window_image_size // 16
    image = np.float32(rescale(image, 1/scale))
    x = np.array(np.round(x,0),dtype=int)//scale
    y = np.array(np.round(y,0),dtype=int)//scale
    
    #이미지를 편미분 합니다.
    Lx = image_diffx(image)
    Ly = image_diffy(image)

    #이미지 전체의 그레디언트의 magnitude, orientation를 구합니다.
    magnitude = np.sqrt(Lx ** 2 + Ly ** 2) 
    orientation = np.arctan2(Ly,Lx) + np.pi

    #interest_point를 기준으로 16X16 windows를 만든다. 
    window_size = 16
    features = []
    for i in range(len(x)):
        edge = 20
        if edge < y[i] and y[i] < image.shape[0] - edge and edge < x[i] and x[i] < image.shape[1] - edge:
            gaussian_kernel = gaussian_filter(4,16)
            crop_magnitude = crop_image(magnitude,[y[i],x[i]],window_size) * gaussian_kernel
            crop_orientation = crop_image(orientation,[y[i],x[i]],window_size)

            #4*4 sub window 생성후 히스토그램 생성 및 벡터 생성
            vector = []
            for j in range(16):
                center_x = (j % 4) * 4 + 2
                center_y = (j // 4) * 4 + 2
                little_magnitude = crop_image(crop_magnitude,[center_y,center_x],4)
                little_orientation = crop_image(crop_orientation,[center_y,center_x],4)
                vector += make_histogram(little_magnitude,little_orientation,8)
            vector = np.array(vector)

        #크기 정규화및 후처리 -> 광도 불변
        vector /= np.sqrt((vector**2).sum())
        vector = np.clip(vector,0,0.2)
        vector /= np.sqrt((vector**2).sum())

        #특징점들 병합 및 리턴
        features.append(vector)
        
    return np.array(features)


def match_features(im1_features, im2_features):

    #두 벡터 사이에 모든 유클리드 거리를 구합니다. array에서 -> [3,4] 값은 1에서 4번째 2에서 5번째 벡터의 유클리드 거리를 말합니다
    Euclidean_distance_array = []
    for i in range(im1_features.shape[0]):
        Euclidean_distance = np.sqrt(((im2_features - im1_features[i])**2).sum(axis = 1))
        Euclidean_distance_array.append(Euclidean_distance)
    Euclidean_distance_array = np.array(Euclidean_distance_array)

    #최소값 2번째 최소값을 구하고 NNDR값을 구해줍니다. -> 정렬하면 order = n^2_lgn, 그냥 두번째 값을 구하면 order = n^2
    min1 = Euclidean_distance_array.min(axis = 1)
    min1_arg = Euclidean_distance_array.argmin(axis = 1)
    Euclidean_distance_array[[i for i in range(im1_features.shape[0])],min1_arg] = 100 #inf
    min2 = Euclidean_distance_array.min(axis = 1)
    nndr = min1/min2

    #이 두값을 이용하여 match를 결정해줍니다. dis = 0.7, NNDR = 0.75
    matches = []
    confidences = []
    for i in range(im1_features.shape[0]):
        if min1[i] < 0.7 and nndr[i] < 0.75:
            matches.append([i,min1_arg[i]])
            confidences.append(-nndr[i])

    matches = np.array(matches)
    confidences = np.array(confidences)

    #같은 점으로 갈경우를 제외해줍니다. 이는  img1 내애서 몰려있는 부분 즉 NNDR이 높다는 것을 뜻합니다. 하지만 다빠지지 않을 수 있습니다.
    for i in range(matches.shape[0]):
        key = matches[i,1]
        for j in range(matches.shape[0]):
            if matches[j,1] == key and i!=j:
                matches[i,1] = -1
                matches[j,1] = -1
    
    ansm = []
    ansc = []
    for i in range(matches.shape[0]):
        if matches[i,1] != -1:
            ansm.append(matches[i])
            ansc.append(confidences[i])

    return np.array(ansm), np.array(ansc)
