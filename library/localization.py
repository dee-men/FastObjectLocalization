#Load the model and dependencies
import time, os, json
import numpy as np
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt
import pickle
import cv2, numpy as np
from deconv_utils import *
import copy
from library.classifiers.pretrained_vgg16 import PretrainedVGG

#Global Variables
percentile_threshold = 40

#Function to get the bounding box given an image
def bbox(im, model, layer=11, n_neurons=5, kmax=10,class_no=0):
    #Inputs
    # im = Output of Image file read by cv2 - Not Resized
    # model = CNN model
    # layer = Layer from which the image has to be extracted
    # n_neurons = Number of Neurons to use to build the image
    # k_max = Maximum Number of neurons to evaluate
    
    mask, cache = get_localization_mask(im, model, layer, n_neurons, kmax,class_no)
    class_no,original_size = cache
    return get_box_from_mask(mask, original_size)

def get_localization_mask(im, model, layer, n_neurons, kmax, class_no=0 ):
    #Get localization Mask for Image
    
    #Inputs
    # im = Output of Image file read by cv2 - Not Resized
    # model = CNN model
    
    original_size = im.shape
    im = resize_image(im)
    im = process_image(im)
    activs, caches = get_activs(model, im)
    
    if (class_no==0):
        class_no = np.argmax(activs[15][0])
    
    back_grad = get_backgrad(activs, model, class_no, layer, caches)
    
    #Get the Filters of Interest in Sorted Order
    amax = filter_of_intr(activs,back_grad,kmax,layer)
    filter_scores = get_filter_scores(amax, model, im, activs,caches, layer, class_no, percentile_thresh=40,use_blob=True)
    
    #Union Blobs
    sorted_scores = sorted(filter_scores,key=lambda x:-x[1])
    mask = np.zeros(im.shape)
    for k in range(n_neurons):
        i,n_score,blob,xmin,xmax,ymin,ymax=sorted_scores[k]
        mask = (mask+blob)>0
        
    localization_cache = class_no,original_size
    return mask, localization_cache

def get_box_from_mask(mask, original_size=0):
    #To get the co-ordinates of Rectangle around the box
    #Inputs
    # Mask - N X D X W X H
    # Original Size is tuple - N X D_orig X W_orig X H_orig
    
    mask = mask[0]
    flat_mask = np.sum(mask, axis = 0)
    
    col_max = np.max(flat_mask, axis = 0)
    row_max = np.max(flat_mask, axis = 1)
    
    col_idxs = np.where(col_max>0)
    xmin = col_idxs[0][0]
    xmax = col_idxs[0][-1]
    
    row_idxs = np.where(row_max>0)
    ymin = row_idxs[0][0]
    ymax = row_idxs[0][-1]
    
    if (original_size!=0):
        W_orig, H_orig,_ = original_size
        xmin = int(xmin*H_orig/224)
        xmax = int(xmax*H_orig/224)
        ymin = int(ymin*W_orig/224)
        ymax = int(ymax*W_orig/224)

    bbox = (xmin, xmax, ymin, ymax)
    return bbox
    
def visualize(im, bbox_cords):
    xmin, xmax, ymin, ymax = bbox_cords
    im = im.astype(np.uint8)
    new_im = im.copy()
    mask = np.zeros(im.shape)
    mask[ymin:ymax,xmin:xmax] = 1
    new_im = new_im*mask
    new_im = new_im.astype(np.uint8)
    new_im = cv2.cvtColor(new_im, cv2.COLOR_BGR2RGB)
    plt.imshow(new_im)
    
def calculate_area(c):
    xmin,xmax,ymin,ymax = c
    return ((xmax-xmin)*(ymax-ymin))

def calculate_overlap(xL1,xH1, xL2, xH2):
    if(xH1>=xL2):
        if(xH2>=xH1):
            return (xH1-xL2)
        else:
            #Box 2 axis lies inside 1
            return (xH2-xL2)
    else:
        return 0
    
def eval_precision(c1,c2):
    xmin1, xmax1, ymin1, ymax1 = c1
    xmin2, xmax2, ymin2, ymax2 = c2
   
    #Order by X
    if (xmin2>=xmin1):
        x_overlap = calculate_overlap(xmin1,xmax1,xmin2,xmax2)
    else:
        x_overlap = calculate_overlap(xmin2,xmax2,xmin1,xmax1)

    if (ymin2>=ymin1):
        y_overlap = calculate_overlap(ymin1,ymax1,ymin2,ymax2)
    else:
        y_overlap = calculate_overlap(ymin2,ymax2,ymin1,ymax1)
   
    intersection = x_overlap*y_overlap
    union = calculate_area(c1) + calculate_area(c2) - intersection
    
    return float(intersection)/float(union)
    
    
def main():
    #Sample Code for an image
    #Load the Image
    imgf = 'Images/dog.jpg'
    im = cv2.imread(imgf)
    
    #Load the VGG Model and ImageNet Class Mappings
    model = PretrainedVGG(h5_file = 'Data/vgg16_weights.h5')
    CLASSES = pickle.load(open('Data/CLASSES.pkl'))
    
    #Layer from which the image will be segment
    layer = 11
    #No of neurons to evaluate from ranked list
    kmax = 10
    #No of neurons to use
    n_neurons = 5
    
    #Get Bounding Box
    bbox_coords = bbox(im,model,layer,n_neurons,kmax)
    #Visualize the Box on the original image
    visualize(im, bbox_cords)
    return bbox_cords
