# -*- coding: utf-8 -*-
"""Background substraction Emotion Mlabs Interview Project .ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1qjldClaRHyThlaapU7wgNxD-F76vjzIC
"""


from deepface import DeepFace
import cv2
import numpy as np
import os
import time
import pandas as pd

def background_substraction(prev,current):
  return np.sum(np.sum(np.absolute(current-prev)))/(np.shape(current)[0]*np.shape(current)[0]);


df=pd.DataFrame(columns=["Participant","angry","fear","neutral","sad","disguist","happy","surprise"])
each_frame_emotion=pd.DataFrame(columns=["Participant","Frame No","Emotion"])
print(df)
tic=time.time()
path=r'Work/Videos'
for vid in os.listdir(path):

    vid_path=os.path.join(path,vid)
    cap=cv2.VideoCapture(vid_path);
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print(frame_count)
    tac=0
    count=0
    prev=0
    
    #Index and Values of emotions set to zero
    
    index_emotion = {0:"angry",1:"fear",2:"neutral",3:"sad",4:"disguist",5:"happy",6:"surprise"}
    value_emotion = {"angry":0,"fear":0,"neutral":0,"sad":0,"disguist":0,"happy":0,"surprise":0}
    
    while True:
        if count==frame_count-1:
            break
        count=count+1
        _,img=cap.read()
        current=img
        current=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        if(count==1):
          prev = current 

        print(f"background_substraction = {background_substraction(prev,current)}")
        
        if (background_substraction(prev,current)>60):
          try:
              gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
              cascadePath = "haarcascades\haarcascade_frontalface_default.xml"
              faceCascade = cv2.CascadeClassifier(cascadePath);
              obj = DeepFace.analyze(img, actions = ['emotion'])
              #objs = DeepFace.analyze(["img1.jpg", "img2.jpg", "img3.jpg"]) #analyzing multiple faces same time
              toc=time.time()
              print("--------\n")
              print("Time is "+str((toc-tic)/(3600))+" hours")
              print("video NO is "+vid[0:len(vid)-4]+" frame no is "+str(count)+"/"+str(frame_count)+" emotion is "+obj["dominant_emotion"])
              print("--------\n")
              
              value_emotion[obj["dominant_emotion"].lower()]+=1
              
              each_frame=[]
              each_frame=[vid[0:len(vid)-4]]
              each_frame.append(count)
              each_frame.append(obj["dominant_emotion"])
              
              each_frame=np.array(each_frame)
              each_frame=np.reshape(each_frame,(each_frame.shape[0],1))  
              #print(np.shape(each_frame))              
              
              each_frame_emotion_df2=pd.DataFrame(each_frame.transpose(),columns=["Participant","Frame No","Emotion"])
              temp=each_frame_emotion_df2.iat[0,2]
              print(temp)
              
              each_frame_emotion=pd.concat([each_frame_emotion,each_frame_emotion_df2])
              #print(each_frame_emotion)
              each_frame_emotion.to_excel("Each Frame Emotion.xlsx",sheet_name='Sheet_name_1')
              prev=current
          except:
              pass
        
    
    total=0
    
    entry_list=[vid[0:len(vid)-4]]
    for i in range(0,len(index_emotion)):
        total+=value_emotion[index_emotion[i]]
    print(value_emotion)
    
    if total!=0:
        for i in range(0,len(index_emotion)):
            entry_list.append(100*value_emotion[index_emotion[i]]/total)
        print(entry_list)
    #else:
    #    entry_list=[vid[0:len(vid)-4],0,0,0,0,0,0,0]
    
    
    
    entry_list=np.array(entry_list)
    entry_list=np.reshape(entry_list,(entry_list.shape[0],1))  
    print(np.shape(entry_list))
    
    
    df2=pd.DataFrame(entry_list.transpose(),columns=["Participant","angry","fear","neutral","sad","disguist","happy","surprise"])
    print(df2)
    df=pd.concat([df,df2])
    print(df)
    
    
    
    df.to_excel("Average Emotion.xlsx",sheet_name='Sheet_name_1')
    each_frame_emotion.to_excel("Each Frame Emotion.xlsx",sheet_name='Sheet_name_1')
