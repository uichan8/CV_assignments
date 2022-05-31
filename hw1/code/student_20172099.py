import numpy as np
from numpy import pi, exp, sqrt
from skimage import io, img_as_ubyte, img_as_float32
from skimage.transform import rescale

def mid(nplist):
    location = np.size(nplist)//2
    nplist = nplist.reshape(np.size(nplist))

    for i in range(location+1):
        for j in range(i+1,np.size(nplist)):
            if nplist[i] > nplist[j]:
                tem = nplist[i]
                nplist[i] = nplist[j]
                nplist[j] = tem
    return nplist[location]

    

    
def my_imfilter(image, kernel):
    """
    이미지에 필터를 적용해주는 함수

    Inputs
    - image: numpy nd-array of dim (m,n) or (m, n, c)
    - kernel: numpy nd-array of dim (k, l)
    - Returns : filtered_image: numpy nd-array of dim of equal 2D size (m,n) or 3D size (m, n, c)
    - Errors if: filter/kernel has any even dimension -> raise an Exception with a suitable error message.
    """
    
    #케널에 대한 검증 : 짝수이면 오류
    assert kernel.shape[0] % 2 != 0, "error: kernel[0] size must be odd"
    assert kernel.shape[1] % 2 != 0, "error: kernel[1] size must be odd"

    #이미지 정보 추출
    is_color = image.ndim > 2
    if is_color:
        third_shape = image.shape[2]
    else:
        third_shape = 0

    #출력이미지 초기화
    filtered_image = np.zeros(image.shape)

    #제로패딩
    row = kernel.shape[0]//2
    col = kernel.shape[1]//2
    if third_shape == 0:
        image = np.pad(image,[(row,row),(col,col)],'constant', constant_values = 0)
    else:
        image = np.pad(image,[(row,row),(col,col),(0,0)],'constant', constant_values = 0)

    #브로드케스팅 하기위한 전처리
    kernel = kernel.reshape(kernel.shape[0],kernel.shape[1],1)

    #컨볼루션(코레이션) : 이미지에서 슬라이싱 한 후에, 내적한걸 filtered_image에 저장
    for i in range(filtered_image.shape[0]):
        for j in range(filtered_image.shape[1]):
            tem = image[i:i+kernel.shape[0],j:j+kernel.shape[1]] * kernel
            if third_shape == 0:
                filtered_image[i,j] = np.sum(np.sum(tem))
            else:
                filtered_image[i,j] = np.sum(np.sum(tem,axis = 0),axis = 0)
                         

    return filtered_image

def my_medfilter(image, size):
    """
    Inputs
    - image: numpy nd-array of dim (m,n) or (m, n, c)
    - size: kernel size
    Returns
    - filtered_image: numpy nd-array of dim of equal 2D size (m,n) or 3D size (m, n, c)
    """
    #케널에 대한 검증 : 짝수이면 오류
    assert size % 2 != 0, "error: kernel size must be odd"

    #이미지 정보 추출
    is_color = image.ndim > 2
    if is_color:
        third_shape = image.shape[2]
    else:
        third_shape = 0

    #출력이미지 초기화
    filtered_image = np.zeros(image.shape)

    #제로패딩 : np.concat으로 붙이기 (제로패딩 사실 필요없음) 
    padding_size = size//2

    if third_shape == 0:
        image = np.pad(image,[(padding_size,padding_size),(padding_size,padding_size)],'constant', constant_values = 0)
    else:
        image = np.pad(image,[(padding_size,padding_size),(padding_size,padding_size),(0,0)],'constant', constant_values = 0)

    #이미지에서 슬라이싱 한 후에, 중간값 산출 및 반환
    for i in range(filtered_image.shape[0]):
        for j in range(filtered_image.shape[1]):
            tem = image[i:i+size,j:j+size]
            if third_shape != 0:
                tem = np.array([mid(tem[:,:,0]),mid(tem[:,:,1]),mid(tem[:,:,2])])
            else:
                tem = mid(tem)
            filtered_image[i,j] = tem

    return filtered_image



def gen_hybrid_image(image1, image2, cutoff_frequency):
    """
     Inputs:
     - image1 -> The image from which to take the low frequencies.
     - image2 -> The image from which to take the high frequencies.
     - cutoff_frequency -> The standard deviation, in pixels, of the Gaussian
                           blur that will remove high frequencies.

     Task:
     - Use my_imfilter to create 'low_frequencies' and 'high_frequencies'.
     - Combine them to create 'hybrid_image'.
    """

    assert image1.shape[0] == image2.shape[0]
    assert image1.shape[1] == image2.shape[1]
    assert image1.shape[2] == image2.shape[2]

    # Steps:
    # (1) Remove the high frequencies from image1 by blurring it. The amount of
    #     blur that works best will vary with different image pairs
    # generate a 1x(2k+1) gaussian kernel with mean=0 and sigma = s, see https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python
    s, k = cutoff_frequency, cutoff_frequency*2
    probs = np.asarray([exp(-z*z/(2*s*s))/sqrt(2*pi*s*s) for z in range(-k,k+1)], dtype=np.float32)
    kernel = np.outer(probs, probs)

    low_frequencies = np.zeros(image1.shape) 
    low_frequencies = my_imfilter(image1,kernel) # 이미지1 필터링

    # (2) Remove the low frequencies from image2. The easiest way to do this is to
    #     subtract a blurred version of image2 from the original version of image2.
    #     This will give you an image centered at zero with negative values.

    high_frequencies = np.zeros(image1.shape) 
    high_frequencies = image2 - my_imfilter(image2,kernel) #이미지2 필터링해서 원본에서 빼줌
    

    # (3) Combine the high frequencies and low frequencies

    hybrid_image = np.zeros(image1.shape) 
    hybrid_image = (low_frequencies + high_frequencies)/2 #합치기
    hybrid_image = np.clip(hybrid_image,0,1) #클리핑


    # (4) At this point, you need to be aware that values larger than 1.0
    # or less than 0.0 may cause issues in the functions in Python for saving
    # images to disk. These are called in proj1_part2 after the call to 
    # gen_hybrid_image().
    # One option is to clip (also called clamp) all values below 0.0 to 0.0, 
    # and all values larger than 1.0 to 1.0.

    return low_frequencies, high_frequencies, hybrid_image
