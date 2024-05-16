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
        img /= 255
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
        predict = feats>theta.reshape(1,-1)
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

if __name__ == "__main__":
    # Viola-Jones algorithm stencil code 
    # Written by Soochahn Lee for Computer Vision @ Kookmin University

    import csv
    import sys
    import argparse
    import numpy as np
    import scipy.io as scio

    import matplotlib
    import matplotlib.pyplot as plt

    from skimage import io, filters, feature, img_as_float32
    from skimage.transform import rescale
    from skimage.color import rgb2gray

    ### size of haar_like_feature ###
    hlf_sz = (18,18)

    # load positve and negative datasetsi
    # positive data: size n_p x h x w, where n_p = number of positive images and h_i, w_i are width and height of images
    data_pos = load_folder_imgs('../data/pos', hlf_sz)[:1000,:,:]
    # negative data: size n_n x h x w, where n_n = number of negative images and h_i, w_i are width and height of images
    data_neg = load_folder_imgs('../data/neg', hlf_sz)[:1000,:,:]

    # concatenate all images
    n_p = data_pos.shape[0]
    n_n = data_neg.shape[0]
    data = np.row_stack([data_pos, data_neg])
    # create ndarray to store positive/negative labels
    labels = np.row_stack([np.ones([n_p,1]), np.zeros([n_n,1])])
    weights = np.row_stack([np.ones([n_p,1]), np.ones([n_n,1])]).astype(np.float32)

    print(1)
    iimgs = get_integral_imgaes(data)

    # 1. 각 feature 종류마다 가능한 위치/크기 값 x, y, w, h을 shape = (개수 x 4)인 ndarray 형태로 도출하는 함수를 구현하시오.
    print(2)
    feat2h_ps = get_feature_pos_sz_2h(hlf_sz)
    feat2v_ps = get_feature_pos_sz_2v(hlf_sz)
    feat3h_ps = get_feature_pos_sz_3h(hlf_sz)
    feat3v_ps = get_feature_pos_sz_3v(hlf_sz)
    feat4_ps = get_feature_pos_sz_4(hlf_sz)

    # 2. 각 feature 종류별로 feature 값을 모두 계산하는 함수를 구현하시오. 계산된 feature 값들은 shape (n_image x n_feat)의 ndarray 형태로 도출하시오.
    print(3)
    feat2h = compute_features_2h(feat2h_ps, iimgs)
    feat2v = compute_features_2v(feat2v_ps ,iimgs)
    feat3h = compute_features_3h(feat3h_ps ,iimgs)
    feat3v = compute_features_3v(feat3v_ps ,iimgs)
    feat4 = compute_features_4(feat4_ps ,iimgs)

    feat = np.column_stack(([feat2h, feat2v, feat3h, feat3v, feat4]))
    num_feat_per_type = [feat2h.shape[1], feat2v.shape[1], feat3h.shape[1], feat3v.shape[1], feat4.shape[1]]

    print(4)
    get_weak_classifiers(feat, labels, weights)
    #thetas, signs, errors = get_weak_classifiers(feat, labels, weights)