# import libraries here
import cv2
import numpy as np
import purple


def count_blood_cells(image_path):
 
    num_of_purple = purple.countP(image_path)
    blood_cell_count = 0
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    image_ada_bin = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 97, 4)

    obPic =255 - image_ada_bin
    kernel = np.ones((3,3))
    obPic = cv2.dilate(obPic, kernel, iterations=3)
    obPic = cv2.erode(obPic, kernel, iterations=3)
    obPic = cv2.erode(obPic, kernel, iterations=3)

    _, contour, _ = cv2.findContours(obPic, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contour:
        _, size, _ = cv2.minAreaRect(contour)
        width, height = size
        if height > 62 and width > 40 and (height < 120  or width < 150):
            blood_cell_count += 1
    return blood_cell_count - num_of_purple
