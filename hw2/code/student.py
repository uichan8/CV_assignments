import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, feature, img_as_int
from skimage.measure import regionprops


def get_interest_points(image, feature_width):
    '''
    Returns interest points for the input image

    Harris corner detector를 구현하세요. 수업시간에 배운 간단한 형태의 알고리즘만 구현해도 됩니다.
    스케일scale, 방향성orientation 등은 추후에 고민해도 됩니다.
    Implement the Harris corner detector (See Szeliski 4.1.1).
    You do not need to worry about scale invariance or keypoint orientation estimation
    for your Harris corner detector.
    원한다면 다른 종류의 특징점 정합 기법을 구현해도 됩니다.
    You can create additional interest point detector functions (e.g. MSER)
    for extra credit.

    만약 영상의 에지 근처에서 잘못된 듯한 특징점이 도출된다면 에지 근처의 특징점을 억제해 버리는 코드를 추가해도 됩니다.
    If you're finding spurious (false/fake) interest point detections near the boundaries,
    it is safe to simply suppress the gradients / corners near the edges of
    the image.

    유용한 함수: 제시해 드리는 모든 함수를 꼭 활용해야 하는 건 아닙니다만, 이중 일부가 여러분에게 유용할 수 있습니다.
    각 함수는 웹에서 제공되는 documentation을 참고해서 사용해 보세요.
    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. 

        - skimage.feature.peak_local_max (experiment with different min_distance values to get good results)
        - skimage.measure.regionprops


    :입력 인자params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :feature_width:

    :반환값returns:
    :xs: an np array of the x coordinates of the interest points in the image
    :ys: an np array of the y coordinates of the interest points in the image

    :옵션으로 추가할 수 있는 반환값optional returns (may be useful for extra credit portions):
    :confidences: an np array indicating the confidence (strength) of each interest point
    :scale: an np array indicating the scale of each interest point
    :orientation: an np array indicating the orientation of each interest point

    '''

    # TODO: Your implementation here! See block comments and the project webpage for instructions

    # These are placeholders - replace with the coordinates of your interest points!
    xs = np.zeros(1,dtype=int)
    ys = np.zeros(1,dtype=int)
    # End of placeholders

    return xs, ys


def get_features(image, x, y, feature_width):
    '''
        주어진 keypoints에 대한 feature descriptor를 구현하시오.
        
        우선 정규화 된 패치를 로컬 기능으로 사용하는 것이 좋습니다.
        Szeliski 4.1.2 원본 사이트를 참조하여 SIFT descriptor와 유사하게 만드십시오.
        
        구현이 정확히 SIFT descriptor와 일치할 필요는 없습니다. 충족 조건만 만족하면 됩니다.
        충족 조건
        1) 4x4 grid 셀
        2) 각 셀은 8방향으로 기울기의 local 분포에 대한 히스토그램을 가져야한다. 
           4*4 * 8 = 128의 특징을 가진 벡터를 생성하게된다.
        3) 각 특징은 단위 길이로 정규화되어야 한다.
        
        각 기울기 측정이 여러 셀의 여러 방향 bin에 기여하는 보간을 수행 할 필요는 없습니다. 
        논문에서 설명 된 거처럼 단일 기울기 측정은 각 cell내에서 가장 가까운 4개의 cell과 가장 가까운 2개의 방향 bin에
        가중치를 부여합니다. 총 8개. 그러나 이런 유형의 보간이 도움될 것이다.
        
        각 픽셀에서 gradient 방향을 명시적으로 계산할 필요 없습니다. 대신 방향 필터로 필터링 할 수 있습니다.
        ex) 특정 방향으로 엣지에 반응하는 필터
        이러한 방식으로 모든 SIFT와 유사한 기능을 상당히 빠르게 필터링하여 구성 할 수 있습니다.
        
        논문에서 설명 된 것처럼 정규화 -> threshold -> 재 정규화 작업을 수행 할 필요없습니다.
        
        최종 특징 벡터의 각 요소를 1보다 작은 거듭 제곱으로 올리는 것도 도움이 될 수 있습니다.
        
        skimage.filters 에 관한 lib를 사용하는 것이 매우 매우 좋습니다.
       ########################################################################################
       
       흐름:
       1. 각 좌표 x, y를 정수로 만드세요. ( 반올림 추천 )
       2. 이미지에 가우시안 filter를 적용하세요.
       3. 이미지 내의 harris corner point 중심으로 16*16 window 만든다.
       4. 16*16 window안의 4*4 sub window를 구성한다.
       5. 16개(4*4)의 sub window에 속한 픽셀들의 gradient의 magnitude, orientation을 계산합니다.
       6. 히스토그램(bin=8)을 만듭니다.( 각 sub window 마다 magnitude, orientation를 사용하여 히스토그램을 생성) - 참고자료 참조 ! 
         (16*16에서 4*4의 윈도우 16개를 얻게 되니까 16*8 = 128 특징 벡터 생성)  
       7. 생성된 벡터를 정규화 합니다.
        
       참고 자료 :
            국문 : https://bskyvision.com/21
            영문 : https://www.analyticsvidhya.com/blog/2019/10/detailed-guide-powerful-sift-technique-image-matching-python/
            
       ######################################################################################## 
        :params:
        :image: grayscale 이미지 (your choice depending on your implementation)
        :x: array형태로 된 관심 좌표의 x값
        :y: array형태로 된 관심 좌표의 x값
        :feature_width: 픽셀 단위의 local feature 너비.
         
        You can assume that feature_width will be a multiple of 4 (i.e. every cell of your   
            local SIFT-like feature will have an integer width and height).
         If you want to detect and describe features at multiple scales or  
         particular orientations you can add input arguments.
    
        :returns:
        :features: np array of computed features. It should be of size
                [len(x) * feature dimensionality] (for standard SIFT 
                dimensionality is 128)
                
                !! 계산된 feauture의 np array, .shape출력했을 때, [len(x) * 128]이 되는 것이 좋습니다.
    '''

    # TODO: Your implementation here! See block comments and the project webpage for instructions

    
    # These are placeholders - replace with the coordinates of your interest points!
    pass
    features = np.zeros((len(x), 128))
    # End of placeholders
    
    return features


def match_features(im1_features, im2_features):
    '''
    가장 까까운 이웃 거리 비율 테스트를 구현하여 두 이미지의 feature points 간에 일치 항목을 할당하십시오.
    Implements the Nearest Neighbor Distance Ratio Test to assign matches between interest points
    in two images.

    Szeliski. section 4.1.3에 있는 "가장 가까운 거리 비율 테스트(NNDR)" 방정식을 구현하십시오.
    Please implement the "Nearest Neighbor Distance Ratio (NNDR) Test" ,
    Equation 4.18 in Section 4.1.3 of Szeliski.
    
    ######################################################################
    
    1. image1, image2 feature vector 추
    2. 특징 벡터의 Euclidean distance를 구하고 정렬한다.
    3. 가장 작은 거리를 찾는다.
    4. 두 거리의 비율을 계산하고 threshold를 넘는지 확인한다.
    
    ######################################################################
    
    :params:
    :im1_features: an np array of features returned from get_features() for interest points in image1
    :im2_features: an np array of features returned from get_features() for interest points in image2

    :returns:
    :matches: an np array of dimension k x 2 where k is the number of matches. The first
            column is an index into im1_features and the second column is an index into im2_features
    :confidences: an np array with a real valued confidence for each match
    '''

    # TODO: Your implementation here! See block comments and the project webpage for instructions

    # These are placeholders - replace with your matches and confidences!
    pass

    matches     = np.zeros((1,2))
    confidences = np.zeros(1) #평가 함수를 사용하기 위해서 confidence값이 필요합니다.
    # End of placeholders

    return matches, confidences
