import numpy as np
import math
import argparse
import scipy.ndimage
from imageio import imread
from numpy.ma.core import exp
from scipy.constants.constants import pi
from skimage.measure import compare_ssim  #skimage version 0.16
from skimage.measure import compare_psnr  #skimage version 0.16

def get_args():
    parser = argparse.ArgumentParser(
        conflict_handler='resolve',
        description='eg: python3 -img1 file1 -img2 file1 -m 1 -c 0' )
    parser.add_argument('-img1','--image_1',required=True,
                        help='image file_1 URL')
    parser.add_argument('-img2','--image_2',required=True,
                        help='image file_2 URL')
    parser.add_argument('-m','--metric',required=True,type = int,
                        help='metric method\
                                0: PSNR ,1:SSIM ')
    parser.add_argument('-c','--isColored',required=True,type = int,
                        help= 'weather the input img is colored')
    return parser.parse_args()

def RGB2YUV( rgb ):
     
    m = np.array([[ 0.29900, -0.16874,  0.50000],
                 [0.58700, -0.33126, -0.41869],
                 [ 0.11400, 0.50000, -0.08131]])
     
    yuv = np.dot(rgb,m)
    yuv[:,:,1:]+=128.0
    return yuv

def YUV2RGB( yuv ):
      
    m = np.array([[ 1.0, 1.0, 1.0],
                 [-0.000007154783816076815, -0.3441331386566162, 1.7720025777816772],
                 [ 1.4019975662231445, -0.7141380310058594 , 0.00001542569043522235] ])
    
    rgb = np.dot(yuv,m)
    rgb[:,:,0]-=179.45477266423404
    rgb[:,:,1]+=135.45870971679688
    rgb[:,:,2]-=226.8183044444304
    rgb =rgb.clip(0,255)
    return rgb

def readimage(img1,img2,isColored):
    im1_data = imread(img1)
    #im1_data = im1_data[scale[0]:-scale[0],scale[1]:-scale[1]]
    im1_data = im1_data.astype(np.float)
    if(isColored != 0):
        im1_data = RGB2YUV(im1_data)
        im1_data = im1_data[:,:,0]
    im2_data = imread(img2)
    #im2_data = im2_data[scale[0]:-scale[0],scale[1]:-scale[1]]
    im2_data = im2_data.astype(np.float)
    if(isColored != 0):
        im2_data = RGB2YUV(im2_data)
        im2_data = im2_data[:,:,0]
    return [im1_data,im2_data]

def psnr(img1, img2, isColored):
    [im1_data,im2_data] = readimage(img1,img2,isColored)
    #return compare_psnr(im1_data,im2_data)
    diff = im1_data  - im2_data 
    mse = np.mean(diff ** 2)
    return 10 * math.log10(255.0**2/mse)

def ssimnaive(img1, img2,isColored):
    [im1_data,im2_data] = readimage(img1,img2,isColored)
    #Variables for Gaussian kernel definition
    gaussian_kernel_sigma=1.5
    gaussian_kernel_width=11
    gaussian_kernel=np.zeros((gaussian_kernel_width,gaussian_kernel_width))
    
    #Fill Gaussian kernel
    for i in range(gaussian_kernel_width):
        for j in range(gaussian_kernel_width):
            gaussian_kernel[i,j]=\
            (1/(2*pi*(gaussian_kernel_sigma**2)))*\
            exp(-(((i-5)**2)+((j-5)**2))/(2*(gaussian_kernel_sigma**2)))

    #squares of input img
    im1_sq = im1_data**2
    im2_sq = im2_data**2
    im1_im2 =  im1_data * im2_data
    
    #Variances obtained by Gaussian filtering of inputs' squares
    im1_data_sigma = scipy.ndimage.filters.convolve(im1_sq,gaussian_kernel)
    im2_data_sigma = scipy.ndimage.filters.convolve(im2_sq,gaussian_kernel)
    
    #Covariance
    im1_im2_sigma = scipy.ndimage.filters.convolve(im1_im2,gaussian_kernel)
    
    #Centered squares of variances
    im1_data_sigma = im1_data_sigma - im1_sq
    im2_data_sigma = im2_data_sigma - im2_sq
    im1_im2_sigma = im1_im2_sigma - im1_im2
    
    #c1/c2 constants
    #First use: manual fitting
    c_1=6.5025
    c_2=58.5225
    
    #Second use: change k1,k2 & c1,c2 depend on L (width of color map)
    l=255
    k_1=0.01
    c_1=(k_1*l)**2
    k_2=0.03

    #Numerator of SSIM
    num_ssim=(2*im1_im2+c_1)*(2*im1_im2_sigma+c_2)
    #Denominator of SSIM
    den_ssim=(im1_sq+im2_sq+c_1)*\
    (im1_data_sigma+im2_data_sigma+c_2)
    #SSIM
    ssim_map=num_ssim/den_ssim
    index=np.average(ssim_map)

    return index

def ssim(img1, img2,isColored):
    [im1_data,im2_data] = readimage(img1,img2,isColored)
    return compare_ssim(im1_data,im2_data)
def main():
    args = get_args()
    if(args.metric == 0) :
        print(psnr(args.image_1,args.image_2,args.isColored))
    elif(args.metric == 1) :
        print(ssim(args.image_1,args.image_2,args.isColored))

if __name__ == '__main__':
    main()
