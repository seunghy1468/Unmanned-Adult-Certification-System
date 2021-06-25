import dlib          
import numpy as np   
import cv2           
import pandas as pd  

facerec = dlib.face_recognition_model_v1("data/dlib_face_recognition_resnet_model_v1.dat")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('data/shape_predictor_68_face_landmarks.dat')

c=0


def return_euclidean_distance(feature_1, feature_2):
    feature_1 = np.array(feature_1)
    feature_2 = np.array(feature_2)
    dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
    return dist



features_known_arr = []

def face_compare(path_id):
    path_features_known_csv = "data/features_all.csv"
    csv_rd = pd.read_csv(path_features_known_csv, header=None)

    for i in range(csv_rd.shape[0]):
        features_someone_arr = []
        for j in range(0, len(csv_rd.ix[i, :])):
            features_someone_arr.append(csv_rd.ix[i, :][j])
        features_known_arr.append(features_someone_arr)
#    print("Faces in Database：", len(features_known_arr))



    img_rd = cv2.imread(path_id)

    img_gray = cv2.cvtColor(img_rd, cv2.COLOR_RGB2GRAY)

    faces = detector(img_gray)

    font = cv2.FONT_HERSHEY_COMPLEX

    pos_namelist = []
    name_namelist = []

    if len(faces) != 0:
        features_cap_arr = []
        for i in range(len(faces)):
            shape = predictor(img_rd, faces[i])
            features_cap_arr.append(facerec.compute_face_descriptor(img_rd, shape))
        for k in range(len(faces)):
#            print("##### image person", k+1, "#####")
            name_namelist.append("unknown")
            pos_namelist.append(tuple([faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top())/4)]))
            e_distance_list = []
            for i in range(len(features_known_arr)):
                if str(features_known_arr[i][0]) != '0.0':
#                    print("with person", str(i + 1), "the e distance: ", end='')
                    e_distance_tmp = return_euclidean_distance(features_cap_arr[k], features_known_arr[i])
#                    print(e_distance_tmp)
                    e_distance_list.append(e_distance_tmp)
                else:       
                    e_distance_list.append(999999999)
                
            similar_person_num = e_distance_list.index(min(e_distance_list))
#            print("Minimum e distance with person", int(similar_person_num)+1)

            if min(e_distance_list) < 0.4:
                name_namelist[k] = "Person "+str(int(similar_person_num)+1)
#                print("May be person "+str(int(similar_person_num)+1))
#            else:
#                print("Unknown person")



#    print("Faces in image now:", name_namelist, "\n")

    
    if 'unknown' in name_namelist:
        c=0
    else:
        c=1
    return c
    
