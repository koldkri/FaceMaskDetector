from tensorflow.keras.models import load_model
import numpy as np
import cv2


model=load_model("model.h5")
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cap=cv2.VideoCapture(0)

while( True):
    _,img=cap.read()
    img=cv2.flip(img,1,1)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # haarcascade works only on gray images
    faces=face_cascade.detectMultiScale(gray,1.2,4)
    for (x,y,w,h) in faces:
        face_img=img[y:y+h,x:x+w]
        resized=cv2.resize(face_img,(150,150))
        normalized=resized/255.0
        reshaped=np.reshape(normalized,(1,150,150,3))
     #   reshaped = np.vstack([reshaped])
        pred = model.predict(reshaped)
        if pred[0][0]>0.65:
            cv2.putText(img,str(pred[0][0]),(x-10,y-10),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),1)
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        else:
            cv2.putText(img,str(1-pred[0][0]),(x-10,y-10),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)


        # print(cropped.shape)

    cv2.imshow("output",img)
    key = cv2.waitKey(1)
    if key==27:
        break;


# img_path = 'C:\\Users\\dell\\IdeaProjects\\project_opencv\\my_testing\\img4.jpg'    # dog



# check prediction

