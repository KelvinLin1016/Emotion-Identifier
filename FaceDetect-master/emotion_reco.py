# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 16:29:21 2019

@author: Data245Lin
"""

import glob
import numpy as np
import random
import cv2
from imutils import face_utils

import imutils
import time
import dlib
from shutil import copyfile
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
import pickle

class EmotionRecognition:
    def __init__(self):
        self.emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"] #Define emotions
        
        self.selffishface = cv2.face.FisherFaceRecognizer_create() #Initialize fisher face classifier
        self.detector=dlib.get_frontal_face_detector()
        #predictor=dlib.shape_predictor(args["shape_predictor"])
        self.predictor=dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    


#%%prepare
    def prepare(self):
        participants = glob.glob("G:\Data245\Lie_Data\source_emotion\Emotion\\*") #Returns a list of all folders with participant numbers
        for x in participants:
            part = "%s" %x[-4:] #store current participant number
            for sessions in glob.glob("%s\\*" %x): #Store list of sessions for current participant
                for files in glob.glob("%s\\*" %sessions):
                    current_session = files[57:60]
                    file = open(files, 'r')
                    emotion = int(float(file.readline())) #emotions are encoded as a float, readline as float, then convert to integer.
                    sourcefile_emotion = glob.glob("../../Lie_Data/source_images/cohn-kanade-images/%s/%s/*" %(part, current_session))[-1] #get path for last image in sequence, which contains the emotion
                    sourcefile_neutral = glob.glob("G:\Data245\Lie_Data\source_images\cohn-kanade-images\\%s\\%s\\*" %(part, current_session))[0] #do same for neutral image
                    dest_neut = "G:\Data245\Lie_Data\sorted_set\\neutral\\%s" %sourcefile_neutral[62:] #Generate path to put neutral image
                    dest_emot = "G:\Data245\Lie_Data\sorted_set\\%s\\%s" %(self.emotions[emotion], sourcefile_emotion[57:]) #Do same for emotion containing image
                    copyfile(sourcefile_neutral, dest_neut) #Copy file
                    copyfile(sourcefile_emotion, dest_emot) #Copy file
#%%

    def get_files(self,emotion): #Define function to get file list, randomly shuffle it and split 80/20
        files = glob.glob("G:\Data245\Lie_Data\sorted_set\\%s\\*" %emotion)
        random.shuffle(files)
        training = files[:int(len(files)*0.8)] #get first 80% of file list
        prediction = files[-int(len(files)*0.2):] #get last 20% of file list
        return training, prediction            

#%%

    def make_sets(self):
        training_data = []
        training_labels = []
        prediction_data = []
        prediction_labels = []
        for emotion in self.emotions:
            training, prediction = self.get_files(emotion)
            #Append data to training and prediction list, and generate labels 0-7
            for item in training:
                image = cv2.imread(item) #open image
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale
                gray = cv2.resize(gray,(640,490))
                training_data.append(gray) #append image array to training data list
                training_labels.append(self.emotions.index(emotion))
            for item in prediction: #repeat above process for prediction set
                image = cv2.imread(item)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray,(640,490))
                prediction_data.append(gray)
                prediction_labels.append(self.emotions.index(emotion))
        return training_data, training_labels, prediction_data, prediction_labels
    #%%
    def run(self):
        print('Start Training')
        training_data, training_labels, prediction_data, prediction_labels = self.make_sets()
        self.fishface.train(training_data, np.asarray(training_labels))
        print('Training Completed!')
        return self._fishface             
    
    
    
    
    
    
    
    
    
    
    #%%Prepare for landmark emotion detect dataset
    def training_sets(self):
        training_data = []
        training_labels = []
        prediction_data = []
        prediction_labels = []
        for emotion in self.emotions:
            training, prediction = self.get_files(emotion)
            #Append data to training and prediction list, and generate labels 0-7
            for item in training:
                image = cv2.imread(item) #open image
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale
                gray = cv2.resize(gray,(640,490))
                rects=self.detector(gray,0)
                for rect in rects:
                    shape=self.predictor(gray,rect)
                    shape=face_utils.shape_to_np(shape)
                shape-=shape[30]
                training_data.append(shape) #append image array to training data list
                training_labels.append(self.emotions.index(emotion))
            for item in prediction: #repeat above process for prediction set
                image = cv2.imread(item)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray,(640,490))
                rects=self.detector(gray,0)
                for rect in rects:
                    shape=self.predictor(gray,rect)
                    shape=face_utils.shape_to_np(shape)
                shape-=shape[30]
                prediction_data.append(shape)
                prediction_labels.append(self.emotions.index(emotion))
        
        training_data=np.reshape(training_data,(len(training_data),-1))
        prediction_data=np.reshape(prediction_data,(len(prediction_data),-1))
    
        return training_data, training_labels, prediction_data, prediction_labels
#%% Training emotions recognition using SVM
if __name__=="__main__":
    print('[INFO] Preparing Training Dataset..')
    rec=EmotionRecognition()    
    training_data, training_labels, prediction_data, prediction_labels = rec.training_sets()
    
    model1=LogisticRegression()
    model2=SVC(gamma='auto')
    
    clf=model1
    print('[INFO] Start Training...')
    clf.fit(training_data,training_labels)
    print('[INFO] Tranining Complete and Prediction Starts..')
    print('Accuracy={}%'.format(clf.score(prediction_data,prediction_labels)*100))
    filename='emotion_rec_svm_01.sav'
    pickle.dump(clf,open(filename,'wb'))
    
    #%%Load Train model
    loaded_model=pickle.load(open(filename,'rb'))
    
    result=loaded_model.score(prediction_data,prediction_labels)
    print(result)