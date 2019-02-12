import cv2
import numpy as np
import sys

def bgr2rgb(img):
    b,g,r = cv2.split(img)
    return cv2.merge([r,g,b])

def DarkChannel(I, w = 15):
    M, N = I.shape[0:2]
    J_dark = np.zeros((M, N))
    w_pad = int(w/2)
    pad = np.pad(I, ((w_pad, w_pad), (w_pad, w_pad), (0, 0)), mode = 'edge')
    for i, j in np.ndindex(J_dark.shape):
        J_dark[i, j] = np.min(pad[i:i+w, j:j+w, :])
    return J_dark

def AtmLight(I, J_dark, p = 0.001):
    M, N = J_dark.shape
    I_flat = I.reshape(M*N, 3)
    dark_flat = J_dark.ravel()
    idx = (-dark_flat).argsort()[:int(M*N*p)]
    arr = np.take(I_flat, idx, axis = 0)
    A = np.mean(arr, axis = 0)
    #A = np.mean(arr, axis = 0)
    return A.reshape(1,3)

def TransmissionEstimate(I, A, w = 15, omega = 0.95):
    return 1 - omega*DarkChannel(I/A, w)

def Guidedfilter(im, p, r = 200,eps = 1e-06):
    mean_I = cv2.boxFilter(im,cv2.CV_64F,(r,r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F,(r,r))
    mean_Ip = cv2.boxFilter(im*p,cv2.CV_64F,(r,r))
    cov_Ip = mean_Ip - mean_I*mean_p

    mean_II = cv2.boxFilter(im*im,cv2.CV_64F,(r,r))
    var_I   = mean_II - mean_I*mean_I

    a = cov_Ip/(var_I + eps)
    b = mean_p - a*mean_I

    mean_a = cv2.boxFilter(a,cv2.CV_64F,(r,r))
    mean_b = cv2.boxFilter(b,cv2.CV_64F,(r,r))

    q = mean_a*im + mean_b
    return q

def Recover(im, t, A, t0 = 0.1):
    rec = np.zeros(im.shape)
    t = cv2.max(t,t0)

    for ind in range(0,3):
        rec[:,:,ind] = (im[:,:,ind]-A[0,ind])/t + A[0,ind]

    return rec

def dehaze(img_path):
    src = cv2.imread(str(img_path))
    I = src.astype('float64')/255
    src_gray_read = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    src_gray = np.float64(src_gray_read)/255
    dark = DarkChannel(I)
    A = AtmLight(I, dark)
    et = TransmissionEstimate(I, A)
    t = Guidedfilter(src_gray, et)
    J = Recover(I, t, A)
    return J