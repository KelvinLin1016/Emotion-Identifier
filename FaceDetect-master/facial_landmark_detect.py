# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 16:16:16 2019

@author: Data245Lin
"""

#%% Facial landmark detection
from imutils.video import VideoStream
from imutils import face_utils
import argparse
import imutils
import time
import dlib
import cv2
import os
import numpy as np
from sklearn.svm import SVC
from sklearn import model_selection
import pickle




class FacialLandmarkEmoRec:
    
#%% Construct the argument parse and parse the arguments
#ap=argparse.ArgumentParser()
#ap.add_argument("-p","--shape-predictor",required=True,help="path to facial landmark predictor")
#ap.add_argument("-r","--picamera",type=int,default=-1,help="whether or not the Raspberry Pi camera should be used")
#args=vars(ap.parse_args())

#%%initialize dlib's face detecotr (HOG-BASED) and then create the ficial landmark predictor
    def __init__(self):
        print('Loading facial landmark predictor...')
        self.detector=dlib.get_frontal_face_detector()
        #predictor=dlib.shape_predictor(args["shape_predictor"])
        self.predictor=dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        self.emo_rec=self.load_emo_rec()
        
    #%%Emotion 
#    def emo_detect(img):
#        if (img[60][1]>img[48][1])and(img[64][1]>img[54][1])and(img[66][1]>img[62][1]):
#            return 'happy'
#        else:
#            return 'neutral'
    
    #%%initialize the video stream and allow the camera sensor to warmup

    #%%Prepare emotion recognize model
    def load_emo_rec(self):
        self.emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"] #Define emotions
        filename='emotion_rec_svm_01.sav'
        self.emo_rec=pickle.load(open(filename,'rb'))
        return self.emo_rec
    
    
    
    #%% Face orientaition
    #def f_ori(self,shape):
        
    
    
    #%%Loop over the frames from the video stream
    def live_rec(self):
        print('Camera sensor warming up...')
        #vs=VideoStream(usePiCamera=args['picmera']>0).start()
        vs=VideoStream().start()
        time.sleep(2.0)
        while True:
            frame=vs.read()
        
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            gray=cv2.resize(gray,(640,490))
            rects=self.detector(gray,0)
            
            for rect in rects:
                shape=self.predictor(gray,rect)
                shape=face_utils.shape_to_np(shape)
                emotion=self.emo_rec.predict(np.reshape(shape-shape[30],(1,-1)))
                for (x,y) in shape:
                    cv2.circle(frame,(x,y),1,(0,0,255),-1)
                cv2.putText(frame,self.emotions[emotion[0]],(shape[0][0]-20,shape[0][1]),cv2.FONT_HERSHEY_SIMPLEX,1.0,(255,255,255),lineType=cv2.LINE_AA)
                print(emotion)
            cv2.imshow("Frame",frame)
            key=cv2.waitKey(1)&0xff
            
            if key==ord("q"):
                break
        
        cv2.destroyAllWindows()
        vs.stop()
    
    
    #%% image landmark recognition
    def pics_rec(self):
        path='../../Lie_Data/sorted_set/happy'
        frames=os.listdir(path)
        emotions=[]
        c=0
        for pic in frames:
            frame=cv2.imread(os.path.join(path,pic))
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            gray=cv2.resize(gray,(640,490))
            rects=self.detector(gray,0)
            
            for rect in rects:
                shape=self.predictor(gray,rect)
                shape=face_utils.shape_to_np(shape)
                for (x,y) in shape:
                    cv2.circle(frame,(x,y),1,(0,0,255),-1)
                c+=1   
            
            emotion=self.emo_rec.predict(shape)
            emotions.append(emotion)
            #key=cv2.waitKey(1)&0xff
            #if key==ord("q"):
             #   cv2.destroyAllWindows()
        error=0
        error_list=[]
        for i,j in enumerate(emotions):
            if j!='happy':
                error+=1
                error_list.append(i)
        print('Accuracy={}%'.format(1-error/len(emotions)))



#%%
if __name__=="__main__":
    rec=FacialLandmarkEmoRec()
    rec.live_rec()