import cv2, os

wajahDir = 'datawajah'
cam = cv2.VideoCapture(0)
cam.set(3, 512) #Kenapa 3? karna kode 3 mengubah lebar dari ukuran kamera
cam.set(4, 512) #Kenapa 4? karna kode 4 mengubah tinggi dari ukuran kamera

faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyeDetector = cv2.CascadeClassifier('haarcascade_eye.xml')
faceID = input('Masukan Face ID yang akan direkam: ')
print("Arahkan wajah anda kedepan webcam. Tunggu proses pengambilan data wajah selesai..")

ambilData = 1
while True: #looping agar mendeteksi fps terus menerus

    retV, frame = cam.read()
    abuAbu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)        #merubah warna output kamera menjadi abu abu
    faces = faceDetector.detectMultiScale(abuAbu, 1.3, 5)       #frame, scalefactor, dan minNeighbors

    for (x, y, w, h) in faces:
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        namaFile = 'wajah.'+str(faceID)+'.'+str(ambilData)+'.jpg'
        cv2.imwrite(wajahDir+'/'+namaFile,frame)
        ambilData += 1

        roiWarna = frame[y:y+h, x:x+w]
        eyes = eyeDetector.detectMultiScale(roiWarna)
        for (xe, ye, we, he) in eyes:
            cv2.rectangle(roiWarna, (xe, ye),(xe+we, ye+he), (0,255,0), 1)

    # menampilkan output kamera
    cv2.imshow('Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    elif ambilData>10:  #Jika telah menangkap 30 frame otomatis aplikasi berhenti
        break

print("Pengambilan data selesai")
cam.release()
cv2.destroyAllWindows()