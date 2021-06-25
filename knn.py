import numpy as np
import cv2
from matplotlib import pyplot as plt
import pytesseract
import re
from pytesseract import image_to_string
from PIL import Image

face_cascade = cv2.CascadeClassifier(
    'data/haarcascade_frontalface_default.xml')

input_img = 'id/id1.jpg'
image = cv2.imread(input_img)
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


#plt.figure(figsize=(12,8))
#plt.imshow(grayImage, cmap='gray')
#plt.xticks([]), plt.yticks([])
#plt.show()

faces = face_cascade.detectMultiScale(grayImage, 1.3, 5, minSize=(70,70)) #min size로 작은 사진은 필터링

for (a, b, c, d) in faces:
    cv2.rectangle(image, (a, b), (a+c, b+d), (0, 255, 0), 2)
#cv2.imshow("Faces found", image) 
#cv2.waitKey(0)

print(faces) #얼굴 좌표
print(a) # 주민등록증과 면허증을 구분할 얼굴 x좌표

def im_trim (img): #함수로 만든다
    if a > 300: # x좌표가 300을 기준으로 오른쪽에 있을 경우 주민등록증의 생년월일 좌표를 찾음
        x = 177; y = 657; #자르고 싶은 지점의 x좌표와 y좌표 지정
        w = 440; h = 71; #x로부터 width, y로부터 height를 지정
        img_trim = img[y:y+h, x:x+w] #trim한 결과를 img_trim에 담는다
        return img_trim #결과물을 리턴
    elif a < 300:# x좌표가 300을 기준으로 왼쪽에 있을 경우 면허증의 생년월일 좌표를 찾음
        x = 419; y = 647;
        w = 276; h = 42;
        img_trim = img[y:y+h, x:x+w]
        return img_trim

def cleanText(readData): #노이즈를 특수문자로 인식할 경우 삭제
    text = re.sub('[©§{}=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', readData)
    return text    

org_image = cv2.imread(input_img)
trim_image = im_trim(org_image)
gray_trim_image = cv2.cvtColor(trim_image, cv2.COLOR_BGR2GRAY) #가장 성능이 좋은 BGR2GRAY로 사진 추출


#plt.imshow(gray_trim_image, cmap='gray')
#plt.imshow(org_image)
#plt.show()

result_string = []
result=0
result_age=0

trim_number = image_to_string(gray_trim_image)
final_number = cleanText(trim_number)
print(final_number)

result_string = final_number.replace(' ','') #인식한 숫자에서 공백을 없앰
result_number = 10*int(result_string[0]) + int(result_string[1])
result = 10*int(result_string[0]) + int(result_string[1])

def agedetect():
    if int(result_string[0]) < 2:
        result_age = 2019 - 2000 - result
    else:
        result_age = 2019 - 1900 - result
    return result








