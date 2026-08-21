import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread, imresize
from scipy import ndimage

#Global Variables

#Parameters to Process blob
class Param:
  process_conv_size = (4,4)
  num_dilation = 20
  num_erosion = 15

def get_filter_scores2(amax, model, im, activs,caches,layer,  class_no, percentile_thresh=80):
    filter_scores = []
    for zero,i,x,y in amax:
        #print i,(x,y)
        back_grad = deconv(model,activs,caches,layer,(0,i,x,y))
        xmin,xmax,ymin,ymax = find_box(back_grad,percentile_thresh=80)
        if (xmin,xmax,ymin,ymax) == (0,0,0,0):
            continue
        n_score = get_score(im,class_no,model, xmin,xmax,ymin,ymax)
        filter_scores += [[i,n_score,xmin,xmax,ymin,ymax]]
    return filter_scores

def get_filter_scores(amax, model, im, activs,caches,layer,  class_no, percentile_thresh=80,use_blob = False, verbose=False):
  filter_scores = []
  for zero,i,x,y in amax:
      back_grad = deconv(model,activs,caches,layer,(0,i,x,y))
      plt.figure()
      plt.imshow(back_grad[0].transpose(1,2,0))
      if use_blob:
          blob = find_blob(back_grad,percentile_thresh=80)
      else:
          xmin,xmax,ymin,ymax = find_box(back_grad,percentile_thresh=percentile_thresh)
          blob = np.zeros(im.shape)
          blob[0,:,xmin:(xmax+1),ymin:(ymax+1)] = 1
      if np.sum(blob) == 0:
          continue
      n_score = get_score(im,class_no,model, mask = blob)
      if (verbose):
            print "score =" ,n_score
      if use_blob:
        filter_scores += [[i,n_score,blob,0,0,0,0]]
      else:
        filter_scores += [[i,n_score,blob,xmin,xmax,ymin,ymax]]
  return filter_scores

def get_fast_filter_scores(amax, model, im, activs,caches,layer,  class_no, percentile_thresh=80,use_blob = False, verbose=False):
  filter_scores = []
  k=0
  for zero,i,x,y in amax:
      #if np.sum(activs[layer][0,i]>0)
      back_grad = deconv(model,activs,caches,layer,(0,i,x,y))
      if use_blob:
          blob = find_blob(back_grad,percentile_thresh=80)
      else:
          xmin,xmax,ymin,ymax = find_box(back_grad,percentile_thresh=percentile_thresh)
          blob = np.zeros(im.shape)
          blob[0,:,xmin:(xmax+1),ymin:(ymax+1)] = 1
      if np.sum(blob) == 0:
          continue
      n_score = -k
      if verbose:
          print "score =" ,n_score
      if use_blob:
        filter_scores += [[i,n_score,blob,0,0,0,0]]
      else:
        filter_scores += [[i,n_score,blob,xmin,xmax,ymin,ymax]]
      k=k+1
  return filter_scores

def get_backgrad (activs, model, class_no, layer, caches):
    back_grad= np.zeros(activs[15].shape)
    back_grad[0,class_no]=1
    for i in reversed(range(layer,16)):
        back_grad  = (back_grad>0)*back_grad
        back_grad, _ = model.backward(back_grad,caches[i])
    return back_grad

def deconv(model,activs,caches,layer,neuron):
    back_grad = np.zeros(activs[layer].shape)
    back_grad[neuron] = 1
    for i in reversed(range(layer+1)):
        back_grad  = (back_grad>0)*back_grad
        back_grad, _ = model.backward(back_grad,caches[i])
    return back_grad

#Define Function for deconv
def deconv_2(model,activs, caches,layer,neuron,slayer):
    back_grad = np.zeros(activs[slayer].shape)
    print back_grad.shape
    if(len(neuron)==3):
        back_grad[neuron[0],neuron[1]] = 1
    else:
        back_grad[neuron] = 1
    for i in reversed(range(layer,slayer+1)):
        back_grad  = (back_grad>0)*back_grad
        back_grad, _ = model.backward(back_grad,caches[i])
    return back_grad

#Takes the processed image to give activations and caches for entire forward pass
def get_activs(model, im, num_layers=16):
    activ = im
    caches = []
    activs = []
    for i in range(num_layers):
        out,cache = model.forward(activ,start = i, end=i)
        activ = out;
        activs += [activ]
        caches += [cache]
    return activs,caches

###############TO-DO###################
def deconv_batch(model, ims, layer=10):
    #Function to get deconv of a batch of images
    #ims : N X 3 X 224 X 224
    pass
    
##############TO DO###########
#Function to get blob
def get_blob():
    pass
    
#Image Utilities
def load_image(imgf):
    im = imread(imgf)
    im = resize_image(im)
    return process_image(im)

def load_image_cv2(imgf):
    im = cv2.imread(imgf)
    im = resize_image(im)
    return process_image(im)

def resize_image(im):
    return cv2.resize(im, (224, 224)).astype(np.float32)

def process_image(im):
    im[:,:,0] -= 103.939
    im[:,:,1] -= 116.779
    im[:,:,2] -= 123.68
    im = im.transpose((2,0,1))
    im = np.expand_dims(im, axis=0)
    return im

def deprocess_image(img):
    im = img[0].transpose(1,2,0)
    im[:,:,0] += 103.939
    im[:,:,1] += 116.779
    im[:,:,2] += 123.68
    im = im.astype(np.uint8)
    return im

#Takes  input 
def plot_image(im):
    im = deprocess_image(im)
    plt.figure()
    plt.imshow(im)

def plot_image_cv2(im):
    im = deprocess_image(im)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    plt.figure()
    plt.imshow(im)
    #cv.imshow("Image",im)
    
#import matplotlib.pyplot as plt
def grid_plot_activs(act):
    grid = visualize_grid((act).transpose(1,2,3,0))
    plt.imshow(grid.transpose(2,0,1)[0])
    #plt.axis('off')
    plt.gcf().set_size_inches(10, 10)
    plt.show()
    
def find_box(back_grad, percentile_thresh = 40):
    meanimg = np.mean(back_grad[0],axis=0)
    if np.sum(abs(meanimg)) < 1e-16:
        return 0,0,0,0
    thresh = np.percentile(meanimg[meanimg>0],[percentile_thresh])[0]
    meanimg = np.mean(back_grad[0],axis=0)
    threshimg = np.mean(back_grad[0],axis=0)>thresh
    #plt.imshow(np.mean(back_grad[0],axis=0)>thresh)
    idxs = np.where(np.sum(threshimg,axis = 1)>0)[0]
    xmin,xmax = min(idxs),max(idxs)
    idxs = np.where(np.sum(threshimg,axis = 0)>0)[0]
    ymin,ymax = min(idxs),max(idxs)
    return xmin,xmax,ymin,ymax

#Get Score for a given image and class number. If mask != 0, it uses the mask area, else defaults to the area specified by the rectangle specified by (xmin,ymin) (xmax,ymax)
def get_score(im,class_no, model, xmin=0,xmax=0,ymin=0,ymax=0,mask=0):
    #print xmin,xmax,ymin,ymax
    if np.sum(mask )== 0:
        mask = np.zeros(im.shape)
        mask[0,:,xmin:(xmax+1),ymin:(ymax+1)] = 1
    newim = im.copy()*mask
    scores,_ = model.forward(newim)
    return scores[0,class_no]

#Given Activations, Deconv, Layer Number
#k = No of Neurons of Interest
#Returns the filters of interest
def filter_of_intr(activs,back_grad,kmax,layer):
   cum =[]
   tally = []
   tmp1 = activs[layer] *back_grad
   k=0
   while k<kmax:
       index = np.unravel_index(tmp1.argmax(), tmp1.shape)
       #print index[1]
       if index[1] not in tally:
           tally += [index[1]]
           cum+= [index]
           k+=1
       tmp1[index] = -1000
   return cum



def grabCut(im2,xmin,xmax,ymin,ymax):
    h,w = im2.shape[:2]
    mask = np.zeros((h,w),dtype='uint8')
    rect = (ymin,xmin,ymax-ymin,xmax-xmin)
    #rect = (10,10,213,213)
    tmp1 = np.zeros((1, 13 * 5))
    tmp2 = np.zeros((1, 13 * 5))  
    cv2.grabCut(im2,mask,rect,tmp1,tmp2,10,mode=cv2.GC_INIT_WITH_RECT)
    mask[mask==2]=0
    return mask


def process_blob(cim):
    #cim = ndimage.binary_erosion(cim>0)
    for i in range(4):
        cim = ndimage.binary_erosion(cim>0)
        cim=ndimage.binary_dilation(cim>0)

    filterk = np.ones(Param.process_conv_size);
    cim = ndimage.convolve(cim, filterk, mode='constant', cval=0.0)

    for i in range(Param.num_dilation):
        cim = ndimage.binary_dilation(cim>0)
    for i in range(Param.num_erosion):
        cim = ndimage.binary_erosion(cim>0)
    return cim

def find_blob(back_grad, percentile_thresh = 40):
    meanimg = np.mean(back_grad[0],axis=0)
    if np.sum(abs(meanimg)) < 1e-16:
        return 0,0,0,0
    thresh = np.percentile(meanimg[meanimg>0],[percentile_thresh])[0]
    meanimg = np.mean(back_grad[0],axis=0)
    threshimg = np.mean(back_grad[0],axis=0)>thresh
    
    return process_blob(threshimg)
