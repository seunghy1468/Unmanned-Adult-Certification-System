import numpy as np
import cv2
from matplotlib import pyplot as plt
import pytesseract
from pytesseract import image_to_string
from PIL import Image

face_cascade = cv2.CascadeClassifier(
    'data/haarcascade_frontalface_default.xml')

input_img = 'id/id1.jpg'
image = cv2.imread(input_img)
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


plt.figure(figsize=(12,8))
#plt.imshow(grayImage, cmap='gray')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
#plt.show()

faces = face_cascade.detectMultiScale(grayImage, 1.3, 5)

for (a, b, c, d) in faces:
    cv2.rectangle(image, (a, b), (a+c, b+d), (0, 255, 0), 2)
#cv2.imshow("Faces found", image)
#cv2.waitKey(0)

print(faces) #얼굴 좌표
print(a) #얼굴 x좌표

def im_trim (img): #함수로 만든다
    if a > 300:
        x = 177; y = 657; #자르고 싶은 지점의 x좌표와 y좌표 지정
        w = 440; h = 71; #x로부터 width, y로부터 height를 지정
        img_trim = img[y:y+h, x:x+w] #trim한 결과를 img_trim에 담는다
        return img_trim #필요에 따라 결과물을 리턴'
    elif a < 300:
        x = 419; y = 647;
        w = 276; h = 42;
        img_trim = img[y:y+h, x:x+w]
        return img_trim
    

org_image = cv2.imread(input_img)
trim_image = im_trim(org_image)


#plt.imshow(trim_image)
#plt.imshow(org_image,cmap = 'gray')
#plt.show()

result_string = []
result=0
result_age=0

print(image_to_string(trim_image))
result_string = image_to_string(trim_image).replace(' ','')
result = 10*int(result_string[0]) + int(result_string[1])

def agedetect():
    if int(result_string[0]) < 2:
        result_age = 2019 - 2000 - result
    else:
        result_age = 2019 - 1900 - result
    return result
