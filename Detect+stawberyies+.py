
# coding: utf-8

# In[43]:

from _future_ import divison
import cv2
from matplotlib import pyplot as plt
import numpy as np
from math import sin,cos

green = (0,255,0)

def show(image):
    plt.figure(figsize = (10,10))
    plt.imshow(image, interpolation='nearest')


def overlay_mask(mask,image):
    rgb_mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2RGB)
    img = cv2.addWeighted(rgb_mask, 0.5,image, 0.5, 0)
    return img

def find_biggest_contour(image):
    # copy image to make changes
    image = image.copy()
    contours,hierarchy = cv2.findContours(image,cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # get all contour area and store in contour size array
    # isolatinag largest contour
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    print (counter_sizes, counter)
    
    biggest_contour = max(contour_sizes, key= lambda x: x[0])[1]
    # return biggest contour
    mask = np.zeros(image.shape, np.unit8)
    cv2.drawContours(mask,[biggest_contour], -1,-255,-1)
    return biggest_contour,mask

def circle_contour(image,contour):
    image_with_ellipse = image.copy()
    ellipse = cv2.fitEllipse(contour)
    cv2.ellipse(image_with_ellipse,ellipse, green, 2, cv2.CV_AA)
    return image_with_ellipse 

def find_stawberry(image): #rgb is better than bgr

    # convert to perfect color scheme 
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    max_dimension = max(image.shape)
    # image is max window size
    # scale our image properly
    # make it in sqaire shape so fx, fy = scale

    scale = 700/max_dimension
    image = cv2.resize(image, None, fx=scale, fy = scale)
    
    # clean the image, make reduce noise from it =, make smooth, blur
    # gaussian funtion helps tomake it smooth
    image_blur = cv2.GaussianBlur(image,(7,7),0)
    # hsv separetes brighness from color,focus on color
    image_blur_hsv = cv2.cvtColor(image_blur, cv2.COLOR_RGB2HSV)
    
    #define filters, filter by color, need specific range to define red 
    min_red= np.array([0,100,80])
    max_red = np.array([10,255,255])
    
    #mask isfilter to focus on one color and black out other 
    mask1 = cv2.inRange(image_blur_hsv, min_red, max_red)
    
    #Brightness 
    min_red2 = np.array([170,100,80])
    max_red2 = np.array([180,256,256])
    
    mask2 = cv2.inRange(image_blur_hsv, min_red2, max_red2)
    
    # combine our mask
    
    mask = mask1 + mask2
    
    # segmentation to separate stawberry from other imgae
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
    # fiting our ellipse around stawberry
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask_clean = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)
    # first perform clossing makes small noice cancelletion, make smooth
    # open is opposite to close, they add to each other
    
    # find biggest stawberry
    # this contour finds bigggest contour and biggest strawberry
    big_strawberry_contour, mask_strawberries = find_biggest_contour(mask_clean)
    
    # overlay this mask on image
    overlay = overlay_mask(mask_clean, image)
    
    # circle the biggest strawberry
    circled = circle_contour(ovelay, big_strawberry_contour)
    show(circled)
    
    # convertback to original color scheme 
    bgr = cv2.cvtColor(circled, cv2.COLOR_RGB2BGR)
    return bgr
    
image = cv2.imread('download.jpg')
result = find_stawberry(image)
cv2.imwrite( 'download2.jpg,result')


# In[ ]:



