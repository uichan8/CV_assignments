import numpy as np
import matplotlib
from skimage.io import imread
from skimage.color import rgb2grey
from skimage.feature import hog, corner_harris, corner_peaks
from skimage.filters import gaussian
from skimage.transform import resize
from scipy.spatial.distance import cdist
from scipy.stats import mode
from sklearn.svm import LinearSVC
from sklearn.cluster import KMeans
from scipy.spatial import distance
from tqdm import tqdm

#전역변수 카테고리 선언
categories1 = {'Kitchen':1, 'Store':2, 'Bedroom':3, 'LivingRoom':4, 'Office':5,'Industrial':6, 'Suburb':7, 'InsideCity':8, 'TallBuilding':9, 'Street':10,'Highway':11, 'OpenCountry':12, 'Coast':13, 'Mountain':14, 'Forest':15}
categories2 = {v:k for k, v in categories1.items()}

def crop_image(image,center:list,window_size):
    """
    이미지를 중심을 기준으로 윈도우를 크롭합니다.
        input:
            center : list [y,x]
    """
    #call by reference, return reference로 바꿀 수 있는지 확인하기
    from_center = window_size//2
    return image[center[0]-from_center:center[0]+window_size-from_center,center[1]-from_center:center[1]+window_size-from_center]

def make_hog_vector(image):
    """hog 벡터만들기"""
    #interest_point = corner_peaks(corner_harris(image), min_distance=10)
    ppc = 16
    cpb = 4
    ori = 9
    image = gaussian(image,0.5)
    hog_vector = hog(image,orientations=ori, pixels_per_cell=(ppc,ppc),cells_per_block=(cpb, cpb),feature_vector=False) # 야매
    hog_vector = hog_vector.reshape(-1, cpb**2*ori)
    return hog_vector

def category_converter(category):
    """ 이름이 들어오면 숫자로, 숫자가 들어오면 이름으로 출력"""
    if type(category)==int:
        return categories2[category]
    return categories1[category]

def get_tiny_images(image_paths):
    '''
    This feature is inspired by the simple tiny images used as features in
    80 million tiny images

    Inputs:
        ima. Eage_paths: a 1-D Python list of stringsch string is a complete
                     path to an image on the filesystem.
    Outputs:
        An n x d numpy array where n is the number of images and d is the
        length of the tiny image representation vector. e.g. if the images
        are resized to 16x16, then d is 16 * 16 = 256.
    '''

    #가우시안 블러를 적용하고 16x16으로 리사이징 한 후에 1렬로 나열합니다.
    resized_image_list = []
    for path in tqdm(image_paths):
        resized_image = imread(path, as_gray= True)
        resized_image = gaussian(resized_image,0.5)
        resized_image = resize(resized_image,(16,16),anti_aliasing_sigma=0.5)
        resized_image = resized_image.reshape(-1)
        resized_image_list.append(resized_image)
    
    #평균을 0 단위벡터화 시키는 작업인데 -> k = 5이상에서는 정확도가 올라가는데 그 아래에서는 오히려 쪼금 떨어집니다.
    resized_image_list = np.array(resized_image_list)
    resized_image_list -= np.mean(resized_image_list)
    resized_image_list /= np.linalg.norm(resized_image_list)

    return resized_image_list

def build_vocabulary(image_paths, vocab_size):
    '''
    This function should sample HOG descriptors from the training images,
    cluster them with kmeans, and then return the cluster centers.

    Inputs:
        image_paths: a Python list of image path strings
         vocab_size: an integer indicating the number of words desired for the
                     bag of words vocab set

    Outputs:
        a vocab_size x (z*z*9) (see below) array which contains the cluster
        centers that result from the K Means clustering.


    '''
    raw_voca = np.array([]).reshape(0,144) #파라미터 주의
    for path in tqdm(image_paths):
        image = imread(path, as_gray= True)
        hog_vector = make_hog_vector(image)
        raw_voca = np.vstack((raw_voca,hog_vector))
    print("k means cluster is running please wait!")
    kmeans = KMeans(n_clusters=vocab_size).fit(raw_voca)
    centers = kmeans.cluster_centers_
        
    return centers

def get_bags_of_words(image_paths):
    '''
    This function should take in a list of image paths and calculate a bag of
    words histogram for each image, then return those histograms in an array.

    Inputs:
        image_paths: A Python list of strings, where each string is a complete
                     path to one image on the disk.

    Outputs:
        An nxd numpy matrix, where n is the number of images in image_paths and
        d is size of the histogram built for each image.
    '''

    vocab = np.load('vocab.npy')
    print('Loaded vocab from file.')

    voca_size = vocab.shape[0]

    vector = []
    for path in tqdm(image_paths):
        image = imread(path, as_gray= True)
        hog_vector = make_hog_vector(image)
        distances = distance.cdist(hog_vector, vocab, 'cosine') #'cosine' 'euclidean'
        distances_arg = np.argmin(distances, axis = 1 )
        histogram = np.zeros(voca_size)
        for i in range(hog_vector.shape[0]):
            histogram[distances_arg[i]] += 1
        histogram /= np.linalg.norm(histogram)
        vector.append(histogram)
        
        
    return np.array(vector)

def svm_classify(train_image_feats, train_labels, test_image_feats):
    '''
    This function will predict a category for every test image by training
    15 many-versus-one linear SVM classifiers on the training data, then
    using those lear
    ned classifiers on the testing data.

    Inputs:
        train_image_feats: An nxd numpy array, where n is the number of training
                           examples, and d is the image descriptor vector size.
        train_labels: An nx1 Python list containing the corresponding ground
                      truth labels for the training data.
        test_image_feats: An mxd numpy array, where m is the number of test
                          images and d is the image descriptor vector size.

    Outputs:
        An mx1 numpy array of strings, where each string is the predicted label
        for the corresponding image in test_image_feats
    '''
    #그냥 다 트레인 셋으로 쓰겠습니다.
    #카테고리를 정수에 할당합니다. 만약 카테고리를 추가하고 싶으면 전역변수 categoies1을 변경해야 합니다.
    #히스토그램 쓸라면 cos distance를 쓰는것도 좋은데 일단은 유클리드 거리.
    train_labels_int = []
    for i in train_labels:
        train_labels_int.append(category_converter(i))
    train_labels_int = np.array(train_labels_int)
    
    model = LinearSVC()
    x = train_image_feats
    y = train_labels_int
    model.fit(x,y)
    svm_labels = model.predict(test_image_feats)
    test_label = []
    for i in list(svm_labels):
        test_label.append(category_converter(int(i)))
    
    return test_label

def nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats):
    '''
    This function will predict the category for every test image by finding
    the training image with most similar features.

    Inputs:
        train_image_feats: An nxd numpy array, where n is the number of training
                           examples, and d is the image descriptor vector size.
        train_labels: An nx1 Python list containing the corresponding ground
                      truth labels for the training data.
        test_image_feats: An mxd numpy array, where m is the number of test
                          images and d is the image descriptor vector size.

    Outputs:s
        An mx1 numpy list of strings, where each string is the predicted label
        for the corresponding image in test_image_feats
    '''

    k = 3

    #카테고리를 정수에 할당합니다. 만약 카테고리를 추가하고 싶으면 전역변수 categoies1을 변경해야 합니다.
    train_labels_int = []
    for i in train_labels:
        train_labels_int.append(category_converter(i))
    train_labels_int = np.array(train_labels_int)

    # Gets the distance between each test image feature and each train image feature
    distances = cdist(test_image_feats, train_image_feats, 'euclidean')

    # 1) Find the k closest features to each test image feature in euclidean space
    sorted_distance = np.argsort(distances)

    # 2) Determine the labels of those k features
    k_nearest_label = sorted_distance[:,:k] #확인 1
    k_nearest_label = train_labels_int[k_nearest_label]

    # 3) Pick the most common label from the k
    k_nearest_label,k_nearest_count = mode(k_nearest_label,axis = 1) #카운트 1째랑 2번째랑 같으면 따로 처리해야됨 근데 k= 1일때도 생각해야됨
    k_nearest_label=k_nearest_label.reshape(-1)

    test_label = []
    for i in list(k_nearest_label):
        test_label.append(category_converter(int(i)))

    # 4) Store that label in a list
    return test_label