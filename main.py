import cv2, os, shutil, dlib
import numpy as np 
import face_compare as fcom
import cvs_maker as cm
import knn

vidcap = cv2.VideoCapture(0)
vidcap.set(3, 480)
image = vidcap.read()
count = 0
x = 0
success = True

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('data/shape_predictor_68_face_landmarks.dat')

age = knn.agedetect()

while success:
  success,image = vidcap.read()
  img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
  faces = detector(img_gray, 0)


  if len(faces) != 0:
    for k, d in enumerate(faces):
      height = (d.bottom() - d.top())
      width = (d.right() - d.left())
      hh = int(height/2)
      ww = int(width/2)
      im_face = np.zeros((int(height*2), width*2, 3), np.uint8)
      
      for ii in range(height*2):
        for jj in range(width*2):
          im_face[ii][jj] = image[d.top()-hh + ii][d.left()-ww + jj]
      
      cv2.imwrite("img/img_face_%d.jpg" % (count), im_face)     # save frame as JPEG file

  count += 1

  if count == 10:
      vidcap.release()
      cv2.destroyAllWindows()
      count += 1
      break

if count == 11:
  cm.cvs_make("img/")
  x = fcom.face_compare('id/id1.jpg')

print(x)
print(age)

if age<19:
  if x:
    print('성인이 아닙니다.')
  else:
    print('본인이 아닙니다.')

else:
  if x:
    print('성인 인증이 되었습니다.')
  else:
    print('본인이 아닙니다.')
