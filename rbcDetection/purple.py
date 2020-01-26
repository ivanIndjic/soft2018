import cv2
import numpy as np

def countP(image_path):
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    image_ada_bin = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 301, 31)
    obPic = 255 - image_ada_bin
    kernel2 = np.ones((3, 3))
    obPic = cv2.dilate(obPic, kernel2, iterations=3)
    obPic = cv2.erode(obPic, kernel2, iterations=3)
    obPic = cv2.dilate(obPic, kernel2, iterations=3)
    _, contour, _ = cv2.findContours(obPic, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    c = []
    for contour in contour:
        _, size, _ = cv2.minAreaRect(contour)
        width, height = size
        if height > 35 and height < 200 and width > 35 and width < 200:
            c.append(contour)

    return len(c)
