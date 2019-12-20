#import cv2 as cv
import math
from PIL import Image
import random as rd

img = Image.open('a01-000u-s00-00.png')
img_name = 'a01-000u-s00-00.png'
name, extension = img_name.split('.')
w, h = img.size
n = math.floor(w / 75)
subimg = (0, 0, 113, 113)
count = 0
img.crop(subimg).save(name + '-' + str(count) + '.' + extension)

for i in range(1, n + 1):
    count += 1
    upper_left = i * 75 + rd.randint(0, 38)
    upper_left = min(upper_left, w - 113)
    subimg = (upper_left, 0, upper_left + 113, 113)
    img.crop(subimg).save(name + '-' + str(count) + '.' + extension)
    if upper_left == w - 113:
        break



#
# cv.imwrite("CS2_copy.jpg", img)
