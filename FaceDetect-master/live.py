# This script will detect faces via your webcam.
# Tested with OpenCV3

import cv2

import random
import emotion_reco as er


# Create the haar casade
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
faceDet_two = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
faceDet_three = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
faceDet_four = cv2.CascadeClassifier("haarcascade_frontalface_alt_tree.xml")


emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]
emo_rec=er.run()
accuracy=er.test_recognizer(emo_rec)
print('Reconizer Accuracy {}'.format(accuracy))
#%%
cap = cv2.VideoCapture(0)#replace 0 with address of video 
while(True):
	# Capture frame-by-frame
    ret, frame = cap.read()
    if not ret: print ('no ret'); continue;

	# Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray,(640,490))
    pred, conf = emo_rec.predict(gray)
    print('{}',format(emotions[pred]))


	#Detect faces in the image
    faces = faceCascade.detectMultiScale(
		gray,
		scaleFactor=1.1,
		minNeighbors=5,
		minSize=(30, 30)
		#flags = cv2.CV_HAAR_SCALE_IMAGE
	)


    print("Found {0} faces!".format(len(faces)))

	# Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame,emotions[pred],(x,h),cv2.FONT_HERSHEY_SIMPLEX,1.0,(255,255,255),lineType=cv2.LINE_AA)


	# Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
