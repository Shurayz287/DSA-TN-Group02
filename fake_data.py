import cv2 
import os
from numpy import random

img = cv2.imread('img/image_00241.jpg')
img_out = 'fake_img/'
# fake zoom random
def fake_crop(image, lech,raito_x, raito_y):
    h, w = image.shape[:2]
    croped_img = image[int(lech * h):int(raito_x * h), int(lech * w):int(raito_y * w)]
    return croped_img
# Hàm fake xoay ảnh theo bao nhiêu do
def fake_xoay(image, do):
    (height, width) = image.shape[:2]
    center = (width // 2, height // 2)
    ronated_img = cv2.warpAffine(image, cv2.getRotationMatrix2D(center, do, 1.0), (width, height))
    return ronated_img
new_name = 'fake_img'
cv2.imwrite(img_out + new_name + '1.jpg', fake_crop(img, random.uniform(0.1, 0.25), random.uniform(0.8, 0.95), random.uniform(0.8, 0.95)))
cv2.imwrite(img_out + new_name + '2.jpg', fake_xoay(img, random.uniform(-30, 30)) )
cv2.imwrite(img_out + new_name + '3.jpg', fake_xoay(fake_crop(img, random.uniform(0.1, 0.25), random.uniform(0.8, 0.95), random.uniform(0.8, 0.95)), random.uniform(-30, 30)) )


