import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import os
import scipy.stats
import helper

def acquire(filename):
    #get the grayscale image and the color image of filename
    orig_img = cv2.imread(os.path.join('imgs', filename))
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    grayscale_img = cv2.cvtColor(orig_img, cv2.COLOR_RGB2GRAY)
    return grayscale_img, orig_img

def preprocess(img):
    #binarize the image and clean up the segments
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    img2 = (img < 220).astype(np.uint8)
    img2 = cv2.erode(img2, se, iterations=2)
    for _ in range(2):
        img2 = cv2.dilate(img2, se, iterations=3)
        img2 = cv2.erode(img2, se, iterations=2)
    return img2

def get_features(contours):
   available_features = helper.features(contours)
   feats = np.zeros([len(contours), 2])
   # loop through all found segments 
   
   for k, contour_features in enumerate(available_features):
       feats[k] = available_features[k]["perimeter"]*0.001, available_features[k]["minor_axis"]*0.001
       
   return feats

def process(filename, class_no):
    img_gray, img = acquire(filename)
    img_preproc = preprocess(img_gray)
    contours, contour_img = helper.compute_contours(img_preproc)
    feats, mu, sig = analyze(contours)
    
    plt.subplots(figsize=(15,5))
    plt.subplot(1,2,1)
    helper.disp_img(contour_img)
    plt.subplot(1,2,2) 
    helper.plot_feats(feats, mu, sig, class_no)
    plt.title('Feature space')
    return feats, mu, sig

def analyze(contours):
    F   = get_features(contours)   
    mu  = np.mean(F, axis = 0) 
    sig = np.cov(F.transpose())
    #print(sig)
    return F, mu, sig

def classify_sample(feats, mu_0, sig_0, mu_1, sig_1):
    pdf_0 = scipy.stats.multivariate_normal(mu_0, sig_0).pdf    
    pdf_1 = scipy.stats.multivariate_normal(mu_1, sig_1).pdf    
    p_feat_given_0 = pdf_0(feats)
    p_feat_given_1 = pdf_1(feats)
    p_0  = 0.5
    p_1 = 0.5
    class_0 = p_feat_given_0 * p_0
    class_1 = p_feat_given_1 * p_1
    if class_0 < class_1:
        class_no = 1
    else:
        class_no = 0
    
    return class_no

def inference(img_gray, img):
    img_preproc = preprocess(img_gray)
    contours, contour_img = helper.compute_contours(img_preproc)
    feats = get_features(contours)
    for contour, feat in zip(contours, feats):
        x, y, w, h = cv2.boundingRect(contour)
        start_point = (x, y)
        end_point = (x+w, y+h)
        class_no = classify_sample(feat, mu_0, sig_0, mu_1, sig_1)
        col = (255,0,0) if class_no == 0 else (0,255,0)
        img = cv2.rectangle(img, start_point, end_point, col, 10)
    return img


if __name__ == "__main__":

    
    feats_0, mu_0, sig_0 = process('sticks.jpg', 0)
    feats_1, mu_1, sig_1 = process('leaves.jpg', 1)

    helper.visualize_model(feats_0, mu_0, sig_0, feats_1, mu_1, sig_1)
    
    img, orig_img = acquire("sticks_leaves.jpg")
    result = inference(img, orig_img)
    plt.subplots(figsize=(12,12))
    helper.disp_img(result)
