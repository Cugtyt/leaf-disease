import numpy as np
import cv2 as cv
from PIL import Image, ImageFilter


def pad_img_RGB(img):
    
    diff = (img.shape[1] - img.shape[0]) // 2
    pad_0 = max(diff, 0)
    pad_1 = max(-diff, 0)
    
    return np.pad(img, ((pad_0, pad_0), (pad_1, pad_1), (0, 0)),  'constant', constant_values=((0,0), (0, 0), (0, 0)))
    
    
def pad_img_gray(img):
    
    diff = (img.shape[1] - img.shape[0]) // 2
    pad_0 = max(diff, 0)
    pad_1 = max(-diff, 0)
    
    return np.pad(img, ((pad_0, pad_0), (pad_1, pad_1)),  'constant', constant_values=((0,0), (0, 0)))


def process_all(img, size=(256, 256)):

    img = np.asarray(img)
    
    # sobel
    x = cv.Sobel(img,cv.CV_16S,1,0)
    y = cv.Sobel(img,cv.CV_16S,0,1)
    absX = cv.convertScaleAbs(x)
    absY = cv.convertScaleAbs(y)
    dst = cv.addWeighted(absX,0.5,absY,0.5,0)

    # img_canny
    img_canny = cv.GaussianBlur(img, (3, 3), 0)
    img_canny = cv.Canny(img_canny, 80, 150)

    # color
    lower_blue = np.array([20,0,0])
    upper_blue = np.array([100,255,255])
    # Threshold the HSV image to get only blue colors
    mask = cv.inRange(img, lower_blue, upper_blue)

    # segment
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    
    # fu diao
    ract = np.random.randint(361)
    imgF = Image.fromarray(pad_img_RGB(img)).rotate(ract).resize(size)
    imgF = imgF.filter(ImageFilter.EMBOSS)
    
    # pingpu 
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    s = np.log1p(np.abs(fshift))
    s[..., 0] = (s[..., 0] - s[..., 0].min()) / (s[..., 0].max() - s[..., 0].min()) * 255
    s[..., 1] = (s[..., 1] - s[..., 1].min()) / (s[..., 1].max() - s[..., 1].min()) * 255
    s[..., 2] = (s[..., 2] - s[..., 2].min()) / (s[..., 2].max() - s[..., 2].min()) * 255
    s = np.asarray(s, dtype=np.uint8)
    
    
    # array to image
    ract = np.random.randint(361)
    img_rgb = Image.fromarray(pad_img_RGB(img)).rotate(ract).resize(size)
    img_sobel = Image.fromarray(pad_img_RGB(dst)).rotate(ract).resize(size)
    img_other = Image.fromarray(np.concatenate(
        (pad_img_gray(img_canny)[..., np.newaxis], pad_img_gray(mask)[..., np.newaxis], pad_img_gray(thresh)[..., np.newaxis]),
        axis=2)).rotate(ract).resize(size)
    img_s = Image.fromarray(pad_img_RGB(s)).rotate(ract).resize(size)
    x_con = np.concatenate((np.asarray(img_rgb), np.asarray(img_sobel), np.asarray(img_other), np.asarray(imgF), np.asarray(img_s)), axis=2)
 
    return x_con
    
