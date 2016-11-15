from __future__ import division
import numpy as np
import scipy
import skimage
import matplotlib.pyplot as plt
from skimage import data, io, filters
from math import sqrt
import itertools
from scipy import signal
import pywt
import time
from PIL import Image
import os
import time

# %matplotlib inline
from IPython import display
import PIL

def wavelet_to_numpy(pywavelet_wavelet):
    approximation_result = []
    details_result = []
    wavelet_levels = len(pywavelet_wavelet)
    
    for x in range(wavelet_levels):
        approximation_result.append(pywavelet_wavelet[x][0])
        details_result.append(pywavelet_wavelet[x][1])
    
    return np.array(approximation_result), np.array(details_result)

def numpy_to_wavelet(approx, details):
    ans_list = []
    for x, y in zip(approx, details):
        ans_list.append((x, (y)))
        
    return ans_list


def del_f2(x, A, orig_im, wave_level, wavelet_name='haar'):
    inter = np.multiply( A , pywt.iswt2(x, wavelet_name) ) - np.multiply(A, orig_im)
    return pywt.swt2(inter, wavelet_name, level=wave_level)


def kill_negative(arr_orig):
    arr = np.array(arr_orig)
    arr[arr < 0] = 0
    return arr


def apg(mask_img, actual_img, lambda_thresh, wave_level, iterations, wavelet_name='haar'):
    
    x_0 = pywt.swt2(actual_img, wavelet_name, level=wave_level, start_level=1)
    x_1 = x_0
    t_0 = 0
    t_1 = 1
    
    for iteration in np.arange(iterations):
        
        approx_x_0, det_x_0 = wavelet_to_numpy(x_0)
        approx_x_1, det_x_1 = wavelet_to_numpy(x_1)
        
        approx_y = approx_x_1 + ((t_0-1)/t_1)*(approx_x_1 - approx_x_0)
        det_y = det_x_1 + ((t_0-1)/t_1)*(det_x_1 - det_x_0)
        
        del_calc = del_f2(numpy_to_wavelet(approx_y, det_y), mask_img, actual_img, wave_level, wavelet_name)
        
        approx_del_calc, det_del_calc = wavelet_to_numpy(del_calc)
        
        approx_g, det_g = approx_y - approx_del_calc, det_y - det_del_calc
        
        x_0 = x_1
        x_1 = numpy_to_wavelet(pywt.threshold(approx_g, lambda_thresh) , pywt.threshold(det_g, lambda_thresh))
        
        t_1_new = (1+ (1+ 4*(t_0**2))**(0.5))/2
        t_0 = t_1
        t_1 = t_1_new
        
        print iteration
        #time.sleep(0.1)
        plt.imshow( kill_negative(pywt.iswt2(x_1, wavelet_name)) , cmap = 'Greys')
        display.clear_output(wait=True)
        display.display(plt.gcf())
        #time.sleep(0.1)
        
    return pywt.iswt2(x_1, wavelet_name)


def switch_black_white(img):
    res = []
    replace_arr = np.arange(256)[::-1]
    orig_arr = np.arange(256)
    for x in img:
        inter = [replace_arr[orig_arr==t][0] for t in x]
        res.append(inter)
        
    return np.array(res)