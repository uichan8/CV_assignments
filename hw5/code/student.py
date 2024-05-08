import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage import filters, feature, img_as_int
from skimage.measure import regionprops
from skimage import io
import os


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
    # iimgs = np.zeros([imgs.shape[0],imgs.shape[1]+1,imgs.shape[2]+1])
    # iimgs[:,1,1] = imgs[:,0,0]

    # for i in range(imgs.shape[1]):
    #     for j in range(imgs.shape[2]):
    #         iimgs[:,i+1,j+1] = imgs[:,i,j] + iimgs[:,i,j+1] + iimgs[:,i+1,j] - iimgs[:,i,j]

    # iimgs = iimgs[:,1:,1:]
    # return integral images
    iimgs = np.cumsum(np.cumsum(imgs, axis=1), axis=2)
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
                    if (i-k)%2 == 0:
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
                    if (j-l)%2 == 0:
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
                    if (i-k)%3 == 0:
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
                    if (j-l)%3 == 0:
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
                    if (i-k)%2 == 0 and (j-l)%2 == 0:
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
    x2 = ps[:,0]+ps[:,2]//2
    x3 = ps[:,0]+ps[:,2]

    y1 = ps[:,1]
    y2 = ps[:,1]+ps[:,3]
    
    feats = np.zeros(iimg.shape[0],ps.shape[0])
    # x1,y1,x2,y2
    feats += iimg[:,x1,y1] - iimg[:,x2,y1] - iimg[:,x1,y2] + iimg[:,x2,y2]

    # x1,y1,x3,y3


    return feats


def compute_features_2v(ps, iimg):
    """
    Compute all positions and sizes of type 2v haar-like-features
    :param ps: an ndarray of all positions x,y and sizes w,h [x,y,w,h] of shape (n_feat x 4)
    :param iimg: an ndarray of integral images of shape (n_img, h_img, w_img)

    :return feats: an ndarray of shape (n_img, n_feat) haar-like feature values for input images
    """
    ### your code here ###
    ps = np.array([])
    return ps


def compute_features_3h(ps, iimg):
    """
    Compute all positions and sizes of type 3h haar-like-features
    :param ps: an ndarray of all positions x,y and sizes w,h [x,y,w,h] of shape (n_feat x 4)
    :param iimg: an ndarray of integral images of shape (n_img, h_img, w_img)

    :return feats: an ndarray of shape (n_img, n_feat) haar-like feature values for input images
    """
    ### your code here ###
    ps = np.array([])
    return ps


def compute_features_3v(ps, iimg):
    """
    Compute all positions and sizes of type 3v haar-like-features
    :param ps: an ndarray of all positions x,y and sizes w,h [x,y,w,h] of shape (n_feat x 4)
    :param iimg: an ndarray of integral images of shape (n_img, h_img, w_img)

    :return feats: an ndarray of shape (n_img, n_feat) haar-like feature values for input images
    """
    ### your code here ###
    ps = np.array([])
    return ps


def compute_features_4(ps, iimg):
    """
    Compute all positions and sizes of type 4 haar-like-features
    :param ps: an ndarray of all positions x,y and sizes w,h [x,y,w,h] of shape (n_feat x 4)
    :param iimg: an ndarray of integral images of shape (n_img, h_img, w_img)

    :return feats: an ndarray of shape (n_img, n_feat) haar-like feature values for input images
    """
    ### your code here ###
    ps = np.array([])
    return ps


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
    num_feat = feats.shape[1]
    thetas = np.array([num_feat])
    signs = np.array([num_feat])
    errors = np.array([num_feat])

    return thetas, signs, errors



def visualize_haar_feature(hlf_sz, x,y,w,h,type):
    """
    Visualize haar-like feature
    :param hlf_sz: tuple (w, h) of size of haar-like feature box, 
    :param x, y, w, h, type: position x,y, size w,h, and type of haar-like feature

    :return hlf_img: image visualizing particular haar-like-feature
    """
    ### your code here ###
    hlf_img = np.ones(hlf_sz)
    
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
    hlf_img = np.ones(hlf_sz)
    return hlf_img
    
