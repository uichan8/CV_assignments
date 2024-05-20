import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage import filters, feature, img_as_int
from skimage.measure import regionprops
from skimage import io
import os
from tqdm import tqdm


def load_folder_imgs(folder_name, img_size):
    """
    Load all images in folder and resize
    :param folder_name: string of path to database root folder
    :param img_size: tuple of size of image (to resize)

    :return data: an ndarray of images of size m x h x w, 
                  where m = number of images and h, w are width and height of images
    """
    ### your code here ###
    # get all image paths in folder and store in list
    path_list = os.listdir(folder_name)
    n = len(path_list)

    # initialize data
    data = np.zeros((n,img_size[0],img_size[1]))

    for i in range(n):
        # load all images from path list
        path = os.path.join(folder_name, path_list[i])
        img = io.imread(path, as_gray=True)
        # resize images to tuple img_size
        img = resize(img, (img_size[0], img_size[1]), anti_aliasing=True)
        # add image to data
        data[i,:,:] = img

    # reture data
    return data

def get_integral_imgaes(imgs):
    """
    Compute integral image for all images in ndarray
    :param imgs: ndarray of images of size m x h x w, 
                 where m = number of images and h, w are width and height of images 

    :return iimgs: an ndarray of integral images of size m x h x w, 
                   where m = number of images and h, w are width and height of images
    """
    ### your code here ###
    iimgs = np.zeros([imgs.shape[0],imgs.shape[1]+1,imgs.shape[2]+1])
    iimgs[:,1,1] = imgs[:,0,0]

    for i in range(imgs.shape[1]):
        for j in range(imgs.shape[2]):
            iimgs[:,i+1,j+1] = imgs[:,i,j] + iimgs[:,i,j+1] + iimgs[:,i+1,j] - iimgs[:,i,j]

    iimgs = iimgs[:,1:,1:]
    #return integral images
    # iimgs = np.cumsum(np.cumsum(imgs, axis=1), axis=2)
    return iimgs

def get_feature_pos_sz_2h(hlf_sz):
    """
    Compute all positions and sizes of type 2h haar-like-features
    :param hlf_sz: basic size of haar-like-feature

    :return ps: an ndarray of all positions x,y and sizes w,h [x,y,w,h] of shape (n_feat x 4)
    """
    ### your code here ###
    img_h, img_w = hlf_sz
    ps = []
    for i in range(img_h):
        for j in range(img_w):
            for k in range(i+1,img_h):
                for l in range(j+1,img_w):
                    if (l-j)%2 == 0 and (l-j) >= 4 and (k-i)>=4:
                        ps.append([i,j,k-i,l-j])
    ps = np.array(ps)
    return ps

def get_feature_pos_sz_2v(hlf_sz):
    """
    Compute all positions and sizes of type 2v haar-like-features
    :param hlf_sz: basic size of haar-like-feature

    :return ps: an ndarray of all positions x,y and sizes w,h [x,y,w,h] of shape (n_feat x 4)
    """
    ### your code here ###
    img_h, img_w = hlf_sz
    ps = []
    for i in range(img_h):
        for j in range(img_w):
            for k in range(i+1,img_h):
                for l in range(j+1,img_w):
                    if (i-k)%2 == 0 and (l-j) >= 4 and (k-i)>=4:
                        ps.append([i,j,k-i,l-j])
    ps = np.array(ps)
    return ps

def get_feature_pos_sz_3h(hlf_sz):
    """
    Compute all positions and sizes of type 3h haar-like-features
    :param hlf_sz: basic size of haar-like-feature

    :return ps: an ndarray of all positions x,y and sizes w,h [x,y,w,h] of shape (n_feat x 4)
    """
    ### your code here ###
    img_h, img_w = hlf_sz
    ps = []
    for i in range(img_h):
        for j in range(img_w):
            for k in range(i+1,img_h):
                for l in range(j+1,img_w):
                    if (j-l)%3 == 0 and (l-j) >= 4 and (k-i)>=4:
                        ps.append([i,j,k-i,l-j])
    ps = np.array(ps)
    return ps

def get_feature_pos_sz_3v(hlf_sz):
    """
    Compute all positions and sizes of type 3v haar-like-features
    :param hlf_sz: basic size of haar-like-feature

    :return ps: an ndarray of all positions x,y and sizes w,h [x,y,w,h] of shape (n_feat x 4)
    """
    ### your code here ###
    img_h, img_w = hlf_sz
    ps = []
    for i in range(img_h):
        for j in range(img_w):
            for k in range(i+1,img_h):
                for l in range(j+1,img_w):
                    if (i-k)%3 == 0 and (l-j) >= 4 and (k-i)>=4:
                        ps.append([i,j,k-i,l-j])
    ps = np.array(ps)
    return ps

def get_feature_pos_sz_4(hlf_sz):
    """
    Compute all positions and sizes of type 4 haar-like-features
    :param hlf_sz: basic size of haar-like-feature

    :return ps: an ndarray of all positions x,y and sizes w,h [x,y,w,h] of shape (n_feat x 4)
    """
    ### your code here ###
    img_h, img_w = hlf_sz
    ps = []
    for i in range(img_h):
        for j in range(img_w):
            for k in range(i+1,img_h):
                for l in range(j+1,img_w):
                    if (i-k)%2 == 0 and (j-l)%2 == 0 and (l-j) >= 4 and (k-i)>=4:
                        ps.append([i,j,k-i,l-j])
    ps = np.array(ps)
    return ps

def compute_features_2h(ps, iimg):
    """
    Compute all positions and sizes of type 2h haar-like-features
    :param ps: an ndarray of all positions x,y and sizes w,h [x,y,w,h] of shape (n_feat x 4)
    :param iimg: an ndarray of integral images of shape (n_img, h_img, w_img)

    :return feats: an ndarray of shape (n_img, n_feat) haar-like feature values for input images
    """
    ### your code here ###
    iimg = np.pad(iimg, ((0,0),(1,0),(1,0)), 'constant',constant_values=0)

    x1 = ps[:,0]
    x2 = ps[:,0]+ps[:,2]

    y1 = ps[:,1]
    y2 = ps[:,1]+ps[:,3]//2
    y3 = ps[:,1]+ps[:,3]

    feats = np.zeros([iimg.shape[0],ps.shape[0]])
    
    feats -= iimg[:,x1,y1] - iimg[:,x2,y1] - iimg[:,x1,y2] + iimg[:,x2,y2]
    feats += iimg[:,x1,y2] - iimg[:,x2,y2] - iimg[:,x1,y3] + iimg[:,x2,y3]

    return feats

def test_compute_features_2h(ps, img):
    rfeature = []
    for i in range(ps.shape[0]):
        x,y,h,w = ps[i]

        if w % 2 != 0:
            print("wrong")

        x1 = x
        x2 = x+h

        y1 = y
        y2 = y+w//2
        y3 = y+w
        
        feature = np.zeros_like(img[0])
        feature[x1:x2,y1:y2] = -1
        feature[x1:x2,y2:y3] = 1
        feature = feature.reshape([1,feature.shape[0],feature.shape[1]])
        rfeature.append((img*feature).sum(axis=(1,2)))
    rfeature = np.array(rfeature)
    return rfeature.T

def compute_features_2v(ps, iimg):
    """
    Compute all positions and sizes of type 2v haar-like-features
    :param ps: an ndarray of all positions x,y and sizes w,h [x,y,w,h] of shape (n_feat x 4)
    :param iimg: an ndarray of integral images of shape (n_img, h_img, w_img)

    :return feats: an ndarray of shape (n_img, n_feat) haar-like feature values for input images
    """
    ### your code here ###
    iimg = np.pad(iimg, ((0,0),(1,0),(1,0)), 'constant',constant_values=0)

    x1 = ps[:,0]
    x2 = ps[:,0]+ps[:,2]//2
    x3 = ps[:,0]+ps[:,2]

    y1 = ps[:,1]
    y2 = ps[:,1]+ps[:,3]
    
    feats = np.zeros([iimg.shape[0],ps.shape[0]])

    feats -= iimg[:,x1,y1] - iimg[:,x1,y2] - iimg[:,x2,y1] + iimg[:,x2,y2]
    feats += iimg[:,x2,y1] - iimg[:,x2,y2] - iimg[:,x3,y1] + iimg[:,x3,y2]
    return feats

def test_compute_features_2v(ps, img):
    rfeature = []
    for i in range(ps.shape[0]):
        x,y,h,w = ps[i]

        if h % 2 != 0:
            print("wrong")

        x1 = x
        x2 = x+h//2
        x3 = x+h

        y1 = y
        y2 = y+w
        
        feature = np.zeros_like(img[0])
        feature[x1:x2,y1:y2] = -1
        feature[x2:x3,y1:y2] = 1
        feature = feature.reshape([1,feature.shape[0],feature.shape[1]])
        rfeature.append((img*feature).sum(axis=(1,2)))
    rfeature = np.array(rfeature)
    return rfeature.T

def compute_features_3h(ps, iimg):
    """
    Compute all positions and sizes of type 3h haar-like-features
    :param ps: an ndarray of all positions x,y and sizes w,h [x,y,w,h] of shape (n_feat x 4)
    :param iimg: an ndarray of integral images of shape (n_img, h_img, w_img)

    :return feats: an ndarray of shape (n_img, n_feat) haar-like feature values for input images
    """
    ### your code here ###
    iimg = np.pad(iimg, ((0,0),(1,0),(1,0)), 'constant',constant_values=0)

    x1 = ps[:,0]
    x2 = ps[:,0]+ps[:,2]

    y1 = ps[:,1]
    y2 = ps[:,1]+ps[:,3]//3
    y3 = ps[:,1]+ps[:,3]*2//3
    y4 = ps[:,1]+ps[:,3]

    feats = np.zeros([iimg.shape[0],ps.shape[0]])
    
    feats -= iimg[:,x1,y1] - iimg[:,x2,y1] - iimg[:,x1,y2] + iimg[:,x2,y2]
    feats += iimg[:,x1,y2] - iimg[:,x2,y2] - iimg[:,x1,y3] + iimg[:,x2,y3]
    feats -= iimg[:,x1,y3] - iimg[:,x2,y3] - iimg[:,x1,y4] + iimg[:,x2,y4]

    feats
    return feats

def test_compute_features_3h(ps, img):
    rfeature = []
    for i in range(ps.shape[0]):
        x,y,h,w = ps[i]

        if w % 3 != 0:
            print("wrong")

        x1 = x
        x2 = x+h

        y1 = y
        y2 = y+w//3
        y3 = y+w*2//3
        y4 = y+w
        
        feature = np.zeros_like(img[0])
        feature[x1:x2,y1:y2] = -1
        feature[x1:x2,y2:y3] = 1
        feature[x1:x2,y3:y4] = -1

        feature = feature.reshape([1,feature.shape[0],feature.shape[1]])
        rfeature.append((img*feature).sum(axis=(1,2)))
    rfeature = np.array(rfeature)
    return rfeature.T

def compute_features_3v(ps, iimg):
    """
    Compute all positions and sizes of type 3v haar-like-features
    :param ps: an ndarray of all positions x,y and sizes w,h [x,y,w,h] of shape (n_feat x 4)
    :param iimg: an ndarray of integral images of shape (n_img, h_img, w_img)

    :return feats: an ndarray of shape (n_img, n_feat) haar-like feature values for input images
    """
    ### your code here ###
    iimg = np.pad(iimg, ((0,0),(1,0),(1,0)), 'constant',constant_values=0)

    x1 = ps[:,0]
    x2 = ps[:,0]+ps[:,2]//3
    x3 = ps[:,0]+ps[:,2]*2//3
    x4 = ps[:,0]+ps[:,2]

    y1 = ps[:,1]
    y2 = ps[:,1]+ps[:,3]

    feats = np.zeros([iimg.shape[0],ps.shape[0]])
    
    feats -= iimg[:,x1,y1] - iimg[:,x2,y1] - iimg[:,x1,y2] + iimg[:,x2,y2]
    feats += iimg[:,x2,y1] - iimg[:,x3,y1] - iimg[:,x2,y2] + iimg[:,x3,y2]
    feats -= iimg[:,x3,y1] - iimg[:,x4,y1] - iimg[:,x3,y2] + iimg[:,x4,y2]
    return feats

def test_compute_features_3v(ps, img):
    rfeature = []
    for i in range(ps.shape[0]):
        x,y,h,w = ps[i]

        if h % 3 != 0:
            print("wrong")

        x1 = x
        x2 = x+h//3
        x3 = x+h*2//3
        x4 = x+h

        y1 = y
        y2 = y+w
        
        feature = np.zeros_like(img[0])
        feature[x1:x2,y1:y2] = -1
        feature[x2:x3,y1:y2] = 1
        feature[x3:x4,y1:y2] = -1

        feature = feature.reshape([1,feature.shape[0],feature.shape[1]])
        rfeature.append((img*feature).sum(axis=(1,2)))
    rfeature = np.array(rfeature)
    return rfeature.T

def compute_features_4(ps, iimg):
    """
    Compute all positions and sizes of type 4 haar-like-features
    :param ps: an ndarray of all positions x,y and sizes w,h [x,y,w,h] of shape (n_feat x 4)
    :param iimg: an ndarray of integral images of shape (n_img, h_img, w_img)

    :return feats: an ndarray of shape (n_img, n_feat) haar-like feature values for input images
    """
    ### your code here ###
    iimg = np.pad(iimg, ((0,0),(1,0),(1,0)), 'constant',constant_values=0)

    x1 = ps[:,0]
    x2 = ps[:,0]+ps[:,2]//2
    x3 = ps[:,0]+ps[:,2]

    y1 = ps[:,1]
    y2 = ps[:,1]+ps[:,3]//2
    y3 = ps[:,1]+ps[:,3]

    feats = np.zeros([iimg.shape[0],ps.shape[0]])
    feats -= iimg[:,x1,y1] - iimg[:,x2,y1] - iimg[:,x1,y2] + iimg[:,x2,y2]
    feats += iimg[:,x2,y1] - iimg[:,x3,y1] - iimg[:,x2,y2] + iimg[:,x3,y2]
    feats += iimg[:,x1,y2] - iimg[:,x2,y2] - iimg[:,x1,y3] + iimg[:,x2,y3]
    feats -= iimg[:,x2,y2] - iimg[:,x3,y2] - iimg[:,x2,y3] + iimg[:,x3,y3]
    return feats

def test_compute_features_4(ps, img):
    rfeature = []
    for i in range(ps.shape[0]):
        x,y,h,w = ps[i]

        if h % 2 != 0 or w % 2 != 0:
            print("wrong")

        x1 = x
        x2 = x+h//2
        x3 = x+h

        y1 = y
        y2 = y+w//2
        y3 = y+w
        
        feature = np.zeros_like(img[0])
        feature[x1:x2,y1:y2] = -1
        feature[x2:x3,y1:y2] = 1
        feature[x1:x2,y2:y3] = 1
        feature[x2:x3,y2:y3] = -1

        feature = feature.reshape([1,feature.shape[0],feature.shape[1]])
        rfeature.append((img*feature).sum(axis=(1,2)))
    rfeature = np.array(rfeature)
    return rfeature.T

def get_weak_classifiers(feats, labels, weights):
    """
    Compute all positions and sizes of type 4 haar-like-features
    :param feats: an ndarray of shape (n_img, n_feat) haar-like feature values for input images
    :param labels: an ndarray of shape (n_img) with pos 1/neg 0 labels of all input imags

    :return thetas: an ndarray of weak classifier threshold values of shape (n_feat)
    :return signs: an ndarray of weak classifier sign values (either +1 or -1) of shape (n_feat) 
    :return errors: an ndarray of weak classifier total error values over all images of shape (n_feat)  
    """
    ### your code here ###
    sorted_feat = np.sort(feats, axis=0)
    thr_list = (sorted_feat[:-1,:] + sorted_feat[1:,:])/2

    num_feat = feats.shape[1]
    thetas = np.zeros([num_feat])
    signs = np.zeros([num_feat])
    errors = np.ones([num_feat]) * weights.max() * feats.shape[0]
    
    for i in tqdm(range(thr_list.shape[0])):
        theta = thr_list[i]
        predict = feats<theta.reshape(1,-1)
        grading = (predict == labels).astype(np.float32)
        acc = np.sum(grading, axis=0)/feats.shape[0]
        sign = ((acc > 0.5)-0.5)*2
        grading = np.abs(grading - (sign==-1).reshape(1,-1))

        grading = (1.-grading) * weights
        error = np.sum(grading, axis=0)

        valid_index = error < errors
        errors[valid_index] = error[valid_index]
        thetas[valid_index] = theta[valid_index]
        signs[valid_index] = sign[valid_index]

    return thetas, signs, errors

def get_best_weak_classifier(errors, num_feat_per_type, feat2h_ps, feat2v_ps, feat3h_ps, feat3v_ps, feat4_ps):
    min_arg = np.argmin(errors)
    feat_label = np.array([0]*num_feat_per_type[0] + [1]*num_feat_per_type[1] + [2]*num_feat_per_type[2] + [3]*num_feat_per_type[3] + [4]*num_feat_per_type[4])
    h_type = feat_label[min_arg]
    feat = np.vstack([feat2h_ps, feat2v_ps, feat3h_ps, feat3v_ps, feat4_ps])
    h_x, h_y, h_h, h_w = feat[min_arg]

    return h_x, h_y, h_w, h_h, h_type

def visualize_haar_feature(x,y,w,h,type,hlf_sz =(18,18)):
    """
    Visualize haar-like feature
    :param hlf_sz: tuple (w, h) of size of haar-like feature box, 
    :param x, y, w, h, type: position x,y, size w,h, and type of haar-like feature

    :return hlf_img: image visualizing particular haar-like-feature
    """
    ### your code here ###
    img = np.zeros(hlf_sz)
    if type == 0:
        img[x:x+h,y:y+w//2] = -1
        img[x:x+h,y+w//2:y+w] = 1

    elif type == 1:
        img[x:x+h//2,y:y+w] = -1
        img[x+h//2:x+h,y:y+w] = 1

    elif type == 2:
        img[x:x+h,y:y+w//3] = -1
        img[x:x+h,y+w//3:y+w*2//3] = 1
        img[x:x+h,y+w*2//3:y+w] = -1
    
    elif type == 3:
        img[x:x+h//3,y:y+w] = -1
        img[x+h//3:x+h*2//3,y:y+w] = 1
        img[x+h*2//3:x+h,y:y+w] = -1
    
    elif type == 4:
        img[x:x+h//2,y:y+w//2] = -1
        img[x:x+h//2,y+w//2:y+w] = 1
        img[x+h//2:x+h,y:y+w//2] = 1
        img[x+h//2:x+h,y+w//2:y+w] = -1

    hlf_img = img
    return hlf_img

def overlay_haar_feature(hlf_sz, x,y,w,h,type, image):
    """
    Visualize haar-like feature
    :param hlf_sz: tuple (w, h) of size of haar-like feature box, 
    :param x, y, w, h, type: position x,y, size w,h, and type of haar-like feature
    :param image: image to overlay haar-like feature

    :return hlf_img: image visualizing particular haar-like-feature
    """
    ### your code here ###
    mask = visualize_haar_feature(x,y,w,h,type)
    hlf_img = image
    hlf_img /= hlf_img.max()
    hlf_img[x:x+h,y:y+w] = 0
    mask = 1.*(mask == 1)
    hlf_img[x:x+h,y:y+w] += mask[x:x+h,y:y+w]
    return hlf_img

class haar_weak_classifier:
    """
    class for haar-like feature-based weak classifier
    :define class attributes(class variables) to include 
      -position, size, type of Haar-like feature
      -threshold and polarity
    :define methods(class functions) as needed
    """
    def __init__(self,feature_mask, theta, weight, sign, h_x, h_y, h_w, h_h, h_type): # <-- add class attributes
        self.feature_mask = feature_mask
        self.theta = theta
        self.weight = weight
        self.sign = sign
        self.h_x = h_x
        self.h_y = h_y
        self.h_w = h_w
        self.h_h = h_h
        self.h_type = h_type

    # add class methods
    ### YOUR CODE HERE ###
    def forward(self, imgs):
        feat = imgs * self.feature_mask[np.newaxis,:,:]
        feat = feat.sum(axis=(1,2))
        result = (feat*self.sign < self.theta*self.sign)*self.weight
        return result
    
    def forward2(self,iimg):
        if self.h_type == 0:
            feat = compute_features_2h(np.array([[self.h_x,self.h_y,self.h_h,self.h_w]]),iimg)
        elif self.h_type == 1:
            feat = compute_features_2v(np.array([[self.h_x,self.h_y,self.h_h,self.h_w]]),iimg)
        elif self.h_type == 2:
            feat = compute_features_3h(np.array([[self.h_x,self.h_y,self.h_h,self.h_w]]),iimg)
        elif self.h_type == 3:
            feat = compute_features_3v(np.array([[self.h_x,self.h_y,self.h_h,self.h_w]]),iimg)
        elif self.h_type == 4:
            feat = compute_features_4(np.array([[self.h_x,self.h_y,self.h_h,self.h_w]]),iimg)
        
        result = ((feat*self.sign < self.theta*self.sign)-0.5)*2*self.weight
        return result

    

def get_strong_classifier(feat, labels, weights,
                          num_feat_per_type, feat2h_ps, feat2v_ps, feat3h_ps, feat3v_ps, feat4_ps,
                          num_weak_cls):
    """
    train strong classifier
    :param feats: an ndarray of shape (n_img, n_feat) haar-like feature values for input images
    :param labels: an ndarray of shape (n_img) with pos 1/neg 0 labels of all input imags
    :param weights: an ndarray of shape (n_img) with weight values of all input imags 
    :param num_feat_per_type: number of features per feature type
    :param feat2h_ps, feat2v_ps, feat3h_ps, feat3v_ps, feat4_ps: ndarry of all positions and sizes of haar-like-features 
    :param num_weak_cls: number of weak classifiers that comprise the strong classifier 

    :return hwc_list: list of haar-like feature based weak classifiers
    """
    hwc_list = []
    ### YOUR CODE HERE ##
    selected_classifiers = []
    for i in range(num_weak_cls):
        #Renormailize weights
        weights /= weights.sum()

        #select best weak classifier
        thetas, signs, errors = get_weak_classifiers(feat, labels, weights)
        for i in selected_classifiers:
            errors[i] = 1e10

        h_x, h_y, h_w, h_h, h_type = get_best_weak_classifier(errors, num_feat_per_type, feat2h_ps, feat2v_ps, feat3h_ps, feat3v_ps, feat4_ps)
        min_arg = np.argmin(errors)
        selected_classifiers.append(min_arg)
        classifier_threshold = thetas[min_arg]
        classifier_sign = signs[min_arg]
        feature_mask = visualize_haar_feature(h_x, h_y, h_w, h_h, h_type)

        #selected weak classifier forward
        target_feature = feat[:,min_arg]
        predict = (target_feature * classifier_sign < classifier_threshold * classifier_sign)  # h(x)
        grading = (predict == labels.flatten()).astype(np.float32) #sigma
        e_ij = (1 - grading)#e_ij
        e_j = (weights.flatten() * e_ij).sum() #e_j
        modified_error = e_j/(1-e_j) #beta
        classifier_weight = -np.log(modified_error) #alpha
        
        #update weights
        weights *= (modified_error ** (1 - e_ij)).reshape([-1,1])

        #add classifier
        clasiifier = haar_weak_classifier(feature_mask, classifier_threshold, classifier_weight, classifier_sign, h_x, h_y, h_w, h_h, h_type)
        hwc_list.append(clasiifier)
    
    return hwc_list


def apply_strong_classifier_training_iimgs(iimgs, hwc_list, thr = 0):
    """
    apply strong classifier to training images
    :param iimgs: training set integral images 

    :return cls_list: list of classification results (classification result = 1 if face, 0 if not)
    """
    ### YOUR CODE HERE ###
    cls_list = []
    for c in hwc_list:
        result = c.forward2(iimgs)
        cls_list.append(result)
    cls_list = np.array(cls_list).sum(axis=0) > thr

    return cls_list


def get_classification_correctnet(labels, cls_list):
    """
    check correctness of classification results
    :param labels: an ndarray of shape (n_img) with pos 1/neg 0 labels of all input imags 
    :param cls_list: an ndarray of shape (n_img) with class estimatations

    :return cls_list: list of True/False results for class estimation input
    """
    ### YOUR CODE HERE ###
    correctness = (labels == cls_list)
    return correctness

def get_incorrect_images(data, correctness_list):
    """
    get incorrect images
    :param data: an ndarray of images of size m x h x w, 
                 where m = number of images and h, w are width and height of images 
    :param correctness_list: list of True/False results for class estimation input

    :return incorrect_imgs: an ndarray of images of size n x h x w, 
                            where n = number of incorrect images and h, w are width and height of images 
    """
    ### YOUR CODE HERE ###
    incorrect_imgs = data[~np.where(correctness_list)[0]]
    return incorrect_imgs


def get_sample_images(path = "example.jpg"):
    """
    :return img: image from path
    """
    ### YOUR CODE HERE ###
    img = np.zeros([0,0])
    img = io.imread("example.png", as_gray=True)
    return img



def detect_face(img, hwc_list, min_scale = 1.0, max_scale = 2.0, num_scales = 3, thr = 0):
    """
    face detection by multi-scale sliding window classification using strong classifier
    :param img: input image
    :param hwc_list: strong classifier compring list of haar-like feature based weak classifiers
    
    :return bboxs: list of bounding boxes of detected faces
    """
    ### YOUR CODE HERE ###
    stride = 2
    bboxes = []
    scale_idx = np.arange(num_scales)
    scales = scale_idx * (max_scale-min_scale)/num_scales + min_scale
    for scale in scales:
        s_img = resize(img, (int(img.shape[0]/scale), int(img.shape[1]/scale)), anti_aliasing=True)
        all_bboxes = []
        all_patches = []
        for i in range(0,s_img.shape[0]-18,stride):
            for j in range(0,s_img.shape[1]-18,stride):
                all_patches.append(s_img[i:i+18,j:j+18])
                all_bboxes.append([int(i*scale),int(j*scale),int(18*scale),int(18*scale)])
        all_patches = np.array(all_patches)
        iall_patches = get_integral_imgaes(all_patches)
        result = apply_strong_classifier_training_iimgs(iall_patches, hwc_list, thr)
        final_bbox = np.array(all_bboxes)[np.where(result)[0]]
        bboxes.append(final_bbox)
    bboxes = np.vstack(bboxes)
    return bboxes


def visualize_bboxes(img, bboxes):
    """
    Visualize bounding boxes
    :param img: input image to overlay bounding boxes
    :param bboxes: bounding boxes

    :return bbox_img: image with overlayed bounding boxes
    """
    ### YOUR CODE HERE ###
    for b in bboxes:
        img[b[0]:b[0]+b[2],b[1]] = 1
        img[b[0]:b[0]+b[2],b[1]+b[3]] = 1
        img[b[0],b[1]:b[1]+b[3]] = 1
        img[b[0]+b[2],b[1]:b[1]+b[3]] = 1
    bbox_img = img
    return bbox_img 