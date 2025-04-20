import cv2
from matplotlib import pyplot as plt
import numpy as np
from imgbeddings import imgbeddings
from PIL import Image
import psycopg2
import os

def take_picture():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cv2.imwrite('captured_image.jpg', frame)
    cap.release()
    return frame

def video_capture():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        cv2.imshow('Video Capture', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()  
    cv2.destroyAllWindows() 
    
alg="haarcascade_frontalface_default.xml"
haar_cascade = cv2.CascadeClassifier(alg)
image = take_picture()
gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
faces = haar_cascade.detectMultiScale(gray_img, scaleFactor=1.05, minNeighbors=2, minSize=(100, 100))

i=0
for (x, y, w, h) in faces:
    cropped_img = gray_img[y:y+h, x:x+w]
    cv2.imwrite(f'Faces/face_{i}.jpg', cropped_img)
    i+=1


# connecting to the database - replace the SERVICE URI with the service URI
conn = psycopg2.connect("dbname=face_recog user=postgres password=Mexic1 host=localhost port=5432")

for filename in os.listdir("Faces"):
    img = Image.open("Faces/" + filename)
    ibed = imgbeddings()
    embedding = ibed.to_embeddings(img)
    print(embedding[0])
    print("hello")
    cur = conn.cursor()
    cur.execute("INSERT INTO pictures values (%s,%s)", (filename, embedding[0].tolist()))
    print(filename)
conn.commit()