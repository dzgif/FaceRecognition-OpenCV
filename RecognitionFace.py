import cv2, os, numpy as np

wajahDir = 'datawajah'
dataDir = 'datawajah'

cam = cv2.VideoCapture(0)
cam.set(3, 512) #Kenapa 3? karna kode 3 mengubah lebar dari ukuran kamera
cam.set(4, 512) #Kenapa 4? karna kode 4 mengubah tinggi dari ukuran kamera

faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faceRecognizer = cv2.face.LBPHFaceRecognizer_create()

faceRecognizer.read(dataDir+'/training.xml')
font = cv2.FONT_HERSHEY_SIMPLEX

id = 0
names = ['Tidak diketahui', 'Fajar', 'DZ']

minWidth = 0.1*cam.get(3)
minHeigth = 0.1*cam.get(4)

while True: #looping agar mendeteksi fps terus menerus

    retV, frame = cam.read()
    frame = cv2.flip(frame, 1) #vertical flip
    # merubah warna output kamera menjadi abu abu
    abuAbu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # frame, scalefactor, dan minNeighbors
    faces = faceDetector.detectMultiScale(abuAbu, 1.2, 5,minSize=(round(minWidth),round(minHeigth)),)

    for (x, y, w, h) in faces:
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        id,confidence = faceRecognizer.predict(abuAbu[y:y+h, x:x+w]) #confidence = 0 artinya cocok
        if confidence<=50 :
            nameID = names[id]
            confidenceTxt = " {0}%".format(round(100-confidence))
        else:
            nameID = names[0]
            confidenceTxt = " {0}%".format(round(100-confidence))
        cv2.putText(frame, str(nameID), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(frame, str(confidenceTxt), (x + 5, y+h-5), font, 1, (255, 255, 0), 1)

    # menampilkan output kamera
    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("EXIT")
cam.release()
cv2.destroyAllWindows()