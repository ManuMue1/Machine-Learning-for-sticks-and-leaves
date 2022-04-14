import numpy as np
from matplotlib import pyplot as plt
import cv2
import scipy.stats

def disp_img(img, title=None, rand_color=False):
    if img.max() == 1 and img.dtype == np.uint8:
        img = img*255
    if rand_color:
        img = img.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR);
        colors = np.random.randint(0, 255, size=(256, 1, 3)).astype(np.uint8)
        colors[0] = 255
        img_col = cv2.LUT(img, colors)
        plt.imshow(img_col)
    else:
        plt.imshow(img, cmap='gray')
    if title:
        plt.title(title)
    plt.axis(False)

def compute_contours(img):
    contours, cntr_img = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    res_img = cv2.cvtColor(255*img, cv2.COLOR_GRAY2RGB)
    res_img = cv2.drawContours(res_img, contours, contourIdx=-1, color=(255, 0, 0), thickness=20)  # colorIdx==-1 : all contours
    return contours, res_img

def features(contours):
    feature_list = []
    for k, c in enumerate(contours):
        # major_axis/minor_axis are the square roots of the eigenvalues of the covariance matrix
        if len(c) > 5:
            (x,y), (minor_axis, major_axis), angle = cv2.fitEllipse(c)
        else:
            (x,y), (minor_axis, major_axis), angle = (0,0), (0,0), 0
        feature_list.append({
            'perimeter':  cv2.arcLength(c, True),
            'area':       cv2.contourArea(c),
            'major_axis': major_axis,
            'minor_axis': minor_axis,
            'ecc':        np.sqrt((major_axis-minor_axis)/major_axis) if major_axis>1e-12 else 0.,
        })
    return feature_list

def plot_feats(feats, mu, sig, class_no):
    col = 'r' if class_no==0 else 'g'
    plt.plot(feats[:, 0], feats[:, 1], col+'s')
    plt.plot(mu[0], mu[1], col+'*', markersize=15)
    sx, sy = np.sqrt(sig[0][0]), np.sqrt(sig[1][1])
    plt.plot([mu[0]-sx, mu[0]+sx], [mu[1], mu[1]], col+':')
    plt.plot([mu[0], mu[0]], [mu[1]-sy, mu[1]+sy], col+':')
    plt.grid(True)

def visualize_model(feats_0, mu_0, sig_0, feats_1, mu_1, sig_1):
    x = np.linspace(0.35, 0.8, 100)
    y = np.linspace(0.14, 0.25, 100)
    X, Y = np.meshgrid(x,y)
    XY = np.dstack((X, Y))
    pdf_0 = scipy.stats.multivariate_normal(mean = mu_0, cov = sig_0).pdf
    pdf_1 = scipy.stats.multivariate_normal(mean = mu_1, cov = sig_1).pdf
    
    fig = plt.figure(figsize=(10, 8))
    plot_feats(feats_0, mu_0, sig_0, 0)
    plot_feats(feats_1, mu_1, sig_1, 1)
    plt.contour(X, Y, pdf_0(XY), cmap='Reds')
    plt.contour(X, Y, pdf_1(XY), cmap='Greens')
    plt.title('Learned model')
    plt.show()
    
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection='3d')
    ax.plot_wireframe(X, Y, pdf_0(XY)+pdf_1(XY), color='k')
    plt.title("Both pdfs")
    plt.show() 
    