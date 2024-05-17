import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, feature, img_as_int
from skimage.measure import regionprops


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
    path_list = []

    n = len(path_list)

    # initialize data
    data = np.zeros((n,img_size[0],img_size[1]))

    # load all images from path list

    # resize images to tuple img_size

    # add image to data

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
    iimgs = np.zeros_like(imgs)

    # return integral images
    return iimgs


def get_feature_pos_sz_2h(hlf_sz):
    """
    Compute all positions and sizes of type 2h haar-like-features
    :param hlf_sz: basic size of haar-like-feature

    :return ps: an ndarray of all positions x,y and sizes w,h [x,y,w,h] of shape (n_feat x 4)
    """
    ### your code here ###
    ps = np.array([])
    return ps


def get_feature_pos_sz_2v(hlf_sz):
    """
    Compute all positions and sizes of type 2v haar-like-features
    :param hlf_sz: basic size of haar-like-feature

    :return ps: an ndarray of all positions x,y and sizes w,h [x,y,w,h] of shape (n_feat x 4)
    """
    ### your code here ###
    ps = np.array([])
    return ps


def get_feature_pos_sz_3h(hlf_sz):
    """
    Compute all positions and sizes of type 3h haar-like-features
    :param hlf_sz: basic size of haar-like-feature

    :return ps: an ndarray of all positions x,y and sizes w,h [x,y,w,h] of shape (n_feat x 4)
    """
    ### your code here ###
    ps = np.array([])
    return ps


def get_feature_pos_sz_3v(hlf_sz):
    """
    Compute all positions and sizes of type 3v haar-like-features
    :param hlf_sz: basic size of haar-like-feature

    :return ps: an ndarray of all positions x,y and sizes w,h [x,y,w,h] of shape (n_feat x 4)
    """
    ### your code here ###
    ps = np.array([])
    return ps


def get_feature_pos_sz_4(hlf_sz):
    """
    Compute all positions and sizes of type 4 haar-like-features
    :param hlf_sz: basic size of haar-like-feature

    :return ps: an ndarray of all positions x,y and sizes w,h [x,y,w,h] of shape (n_feat x 4)
    """
    ### your code here ###
    ps = np.array([])
    return ps



def compute_features_2h(ps, iimg):
    """
    Compute all positions and sizes of type 2h haar-like-features
    :param ps: an ndarray of all positions x,y and sizes w,h [x,y,w,h] of shape (n_feat x 4)
    :param iimg: an ndarray of integral images of shape (n_img, h_img, w_img)

    :return feats: an ndarray of shape (n_img, n_feat) haar-like feature values for input images
    """
    ### your code here ###
    feats = np.array([])
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
    :param weights: an ndarray of shape (n_img) with weight values of all input imags 

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


def get_best_weak_classifier(errors, num_feat_per_type, feat2h_ps, feat2v_ps, feat3h_ps, feat3v_ps, feat4_ps):
    """
    Get Haar-like feature parameters of best weak classifier
    :param errors: error values for all weak classifiers
    :param num_feat_per_type: number of features per feature type
    :param feat2h_ps, feat2v_ps, feat3h_ps, feat3v_ps, feat4_ps: ndarry of all positions and sizes of haar-like-features 

    :return x,y,w,h,type: position, size and type of Haar-like feature of best weak classifier
    """
    ### your code here ###
    x = 0
    y = 0
    w = 0 
    h = 0
    type = 0
    return x, y, w, h, type



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
    

class haar_weak_classifier:
    """
    class for haar-like feature-based weak classifier
    :define class attributes(class variables) to include 
      -position, size, type of Haar-like feature
      -threshold and polarity
    :define methods(class functions) as needed
    """
    def __init__(self): # <-- add class attributes
        pass

    # add class methods
    ### YOUR CODE HERE ###



def get_strong_classifier(feat, labels, weights,\
                          num_feat_per_type, feat2h_ps, feat2v_ps, feat3h_ps, feat3v_ps, feat4_ps,\
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
    ### YOUR CODE HERE ###
    hwc_list.append(haar_weak_classifier()) # <-- modify code

    return hwc_list



def apply_strong_classifier_training_iimgs(iimgs, hwc_list):
    """
    apply strong classifier to training images
    :param iimgs: training set integral images 

    :return cls_list: list of classification results (classification result = 1 if face, 0 if not)
    """
    ### YOUR CODE HERE ###
    cls_list = []

    return cls_list


def get_classification_correctnet(labels, cls_list):
    """
    check correctness of classification results
    :param labels: an ndarray of shape (n_img) with pos 1/neg 0 labels of all input imags 
    :param cls_list: an ndarray of shape (n_img) with class estimatations

    :return cls_list: list of True/False results for class estimation input
    """
    ### YOUR CODE HERE ###
    correctness = []

    return correctness


def get_sample_images(data, spl_idx):
    """
    subsample images
    :param data: input of all images
    :param spl_idx: list of indices to sample
    
    :return sample_imgs: sampled images
    """
    ### YOUR CODE HERE ###
    sample_imgs = []

    return sample_imgs


def detect_face(img, hwc_list, min_scale = 1.0, max_scale = 4.0, num_scales = 9):
    """
    face detection by multi-scale sliding window classification using strong classifier
    :param img: input image
    :param hwc_list: strong classifier compring list of haar-like feature based weak classifiers

    :return bboxs: list of bounding boxes of detected faces
    """
    ### YOUR CODE HERE ###
    bboxes = []
    # *** multi-scale input image, similar to code below
    scale_idx = np.arange(num_scales)
    scales = scale_idx * (max_scale-min_scale)/num_scales + min_scale
    for scale in scales:
        s_img = img.rescale(1/scale) # apply resizing to scale input image, than apply face detection

    return bboxes


def visualize_bboxes(img, bboxes):
    """
    Visualize bounding boxes
    :param img: input image to overlay bounding boxes
    :param bboxes: bounding boxes

    :return bbox_img: image with overlayed bounding boxes
    """
    ### YOUR CODE HERE ###
    bbox_img = np.zeros_like(img)
    return bbox_img 
