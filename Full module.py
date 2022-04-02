# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import cv2
import pandas as pd
import re
from deepface import DeepFace
import os
import time
import dlib
import joblib

def globalScore(emotion_percentage,emotion_thershold):
    '''
    Overall performance of interview by giving the dominant emotion and its corresponding value
    
    Argument:-
    emotion_percentage --- pandas dataframe datatype, It contain the percentage of each emotion
    emotion_thershold --- dictonary datatype, key = emotion and value = thershold of that emotion 
    
    return:-
    global_emotion_score --- dictonary, which contains all dominant emotion and its value
    '''
    
    global_emotion_score={}
    
    for emotion in emotion_percentage.keys():
        if(emotion_percentage[emotion].unique()[0]>emotion_thershold[emotion]):
            global_emotion_score[emotion]=emotion_percentage[emotion].unique()[0]
        
    return global_emotion_score


emotion_thershold = {}
emotion_thershold['happy'] = 20
emotion_thershold['surprise'] = 5
emotion_thershold['disguist'] = 12
emotion_thershold['fear'] = 10
emotion_thershold['neutral'] = 10
emotion_thershold['sad'] = 15
emotion_thershold['angry'] = 4


def TimeOverlap(T1_f1,T2_f1,T1_f2,T2_f2):
    if(T2_f1>T1_f2):
        return True
    else:
        return False
    

def TimeIntervalScoreSad(each_frame_emotion,emotion_thershold,interval_value=75,frame_diff=1):
    '''
    Time interval and corresponding dominanting emotion
    
    Argument:-
    each_frame_emotion --- pandas dataframe datatype, each frame and its corresponding emotion
    emotion_thershold --- dictonary datatype, key = emotion and value = thershold of that emotion
    interval_value --- interval in which pattern is searched
    frame_diff --- differece between two frames under consideration
    
    return:-
    interval_range --- list of dictonary datatype, T1-starting frame, T2-ending frame, emotion-sad ,character-character shown during T1 and T2 ,value- % of that particular character
    '''
    
    interval_range=[]
    count=0
    
    array_emotion = np.array(each_frame_emotion['Emotion'])
    array_frame = np.array(each_frame_emotion['Frame No'])
    now_sad = 0
    
    for i in range((interval_value-1)//frame_diff):
        if(array_emotion[i]=='sad'):
            now_sad = now_sad + 1
        
    right = interval_value//frame_diff -1
    
    for left in range (len(array_emotion)):
        if(right==(len(array_emotion))-1):
            break;
        
        if(array_emotion[left]=='sad'):
            now_sad = now_sad - 1
            
        if(array_emotion[right+1]=='sad'):
            now_sad = now_sad + 1
        
        #print("left={} right={} left*frame_diff={} right*frame_diff={} now_sad={} maximum_sad={}".format(left,right,left*frame_diff,right*frame_diff,now_sad,maximum_sad))
        
        right=right+1
        
        if(emotion_thershold['sad']<now_sad*100/(interval_value//frame_diff)):
            if(count==0 or TimeOverlap(interval_range[count-1]['T1'],interval_range[count-1]['T2'],left*frame_diff,right*frame_diff) != True):
                interval={}
                interval['T1']=int(array_frame[left])
                interval['T2']=int(array_frame[right])
                interval['emotion']='sad'
                interval['value']=now_sad*100/(interval_value//frame_diff)
                interval_range.append(interval)
                count=count+1
            elif(interval_range[count-1]['value']<now_sad*100/(interval_value//frame_diff)):
                interval_range.pop()
                interval['T1']=int(array_frame[left])
                interval['T2']=int(array_frame[right])
                interval['emotion']='sad'
                interval['value']=now_sad*100/(interval_value//frame_diff)
                interval_range.append(interval)
                
        
    return interval_range
    
    
def TimeIntervalScoreFear(each_frame_emotion,emotion_thershold,interval_value=75,frame_diff=1):
    '''
    Time interval and corresponding dominanting emotion
    
    Argument:-
    each_frame_emotion --- pandas dataframe datatype, each frame and its corresponding emotion
    emotion_thershold --- dictonary datatype, key = emotion and value = thershold of that emotion
    interval_value --- interval in which pattern is searched
    frame_diff --- differece between two frames under consideration
    
    return:-
    interval_range --- list of dictonary datatype, T1-starting frame, T2-ending frame, emotion- ,character-character shown during T1 and T2 ,value- % of that particular character
    '''
    
    interval_range=[]
    count=0
    
    array_emotion = np.array(each_frame_emotion['Emotion'])
    array_frame = np.array(each_frame_emotion['Frame No'])
    now_fear = 0
    
    for i in range((interval_value-1)//frame_diff):
        if(array_emotion[i]=='fear'):
            now_fear = now_fear + 1
        
    right = interval_value//frame_diff -1
    
    for left in range (len(array_emotion)):
        if(right==(len(array_emotion))-1):
            break;
        
        if(array_emotion[left]=='fear'):
            now_fear = now_fear - 1
            
        if(array_emotion[right+1]=='fear'):
            now_fear = now_fear + 1
        
        #print("left={} right={} left*frame_diff={} right*frame_diff={} now_fear={} maximum_fear={}".format(left,right,left*frame_diff,right*frame_diff,now_fear,maximum_fear))
        
        right=right+1
        
        if(emotion_thershold['fear']<now_fear*100/(interval_value//frame_diff)):
            if(count==0 or TimeOverlap(interval_range[count-1]['T1'],interval_range[count-1]['T2'],left*frame_diff,right*frame_diff) != True):
                interval={}
                interval['T1']=int(array_frame[left])
                interval['T2']=int(array_frame[right])
                interval['emotion']='fear'
                interval['value']=now_fear*100/(interval_value//frame_diff)
                interval_range.append(interval)
                count=count+1
            elif(interval_range[count-1]['value']<now_fear*100/(interval_value//frame_diff)):
                interval_range.pop()
                interval['T1']=int(array_frame[left])
                interval['T2']=int(array_frame[right])
                interval['emotion']='fear'
                interval['value']=now_fear*100/(interval_value//frame_diff)
                interval_range.append(interval)
                
        
    return interval_range
    
    
def TimeIntervalScoreAngry(each_frame_emotion,emotion_thershold,interval_value=75,frame_diff=1):
    '''
    Time interval and corresponding dominanting emotion
    
    Argument:-
    each_frame_emotion --- pandas dataframe datatype, each frame and its corresponding emotion
    emotion_thershold --- dictonary datatype, key = emotion and value = thershold of that emotion
    interval_value --- interval in which pattern is searched
    frame_diff --- differece between two frames under consideration
    
    return:-
    interval_range --- list of dictonary datatype, T1-starting frame, T2-ending frame, emotion-angry ,character-character shown during T1 and T2 ,value- % of that particular character
    '''
    
    interval_range=[]
    count=0
    
    array_emotion = np.array(each_frame_emotion['Emotion'])
    array_frame = np.array(each_frame_emotion['Frame No'])
    now_angry = 0
    
    for i in range((interval_value-1)//frame_diff):
        if(array_emotion[i]=='angry'):
            now_angry = now_angry + 1
        
    right = interval_value//frame_diff -1
    
    for left in range (len(array_emotion)):
        if(right==(len(array_emotion))-1):
            break;
        
        if(array_emotion[left]=='angry'):
            now_angry = now_angry - 1
            
        if(array_emotion[right+1]=='angry'):
            now_angry = now_angry + 1
        
        #print("left={} right={} left*frame_diff={} right*frame_diff={} now_angry={} maximum_angry={}".format(left,right,left*frame_diff,right*frame_diff,now_angry,maximum_anger))
        
        right=right+1
        
        if(emotion_thershold['angry']<now_angry*100/(interval_value//frame_diff)):
            if(count==0 or TimeOverlap(interval_range[count-1]['T1'],interval_range[count-1]['T2'],left*frame_diff,right*frame_diff) != True):
                interval={}
                interval['T1']=int(array_frame[left])
                interval['T2']=int(array_frame[right])
                interval['emotion']='angry'
                interval['value']=now_angry*100/(interval_value//frame_diff)
                interval_range.append(interval)
                count=count+1
            elif(interval_range[count-1]['value']<now_angry*100/(interval_value//frame_diff)):
                interval_range.pop()
                interval['T1']=int(array_frame[left])
                interval['T2']=int(array_frame[right])
                interval['emotion']='angry'
                interval['value']=now_angry*100/(interval_value//frame_diff)
                interval_range.append(interval)
                
        
    return interval_range

    
    
def ConvertToString(each_frame_emotion,emotion_number):
    
    result_string = ""
    array_emotion = np.array(each_frame_emotion['Emotion'])
    
    for emotion in array_emotion:
        result_string += str(emotion_number[emotion])
    
    result_string +="."
    
    return result_string


emotion_number = {}
emotion_number['angry'] = 0
emotion_number['sad'] = 1
emotion_number['fear'] = 2
emotion_number['disguist'] = 3
emotion_number['neutral'] = 4
emotion_number['surprise'] = 5
emotion_number['happy'] = 6


def MinThershold(t1,t2,frame_diff=1):
    if int(t2)-int(t1)>=frame_diff*300:
        return True
    return False


def FindPattern(each_frame_pd,emotion_number,frame_diff=1):
    '''
    Finding pattern using regular expression
    
    Argument:-
    string --- Each frame convert into string representing emotion
    '''
    
    string=ConvertToString(each_frame_pd,emotion_number)
    array_frame = np.array(each_frame_pd['Frame No'])
    
    all_regex = []
    character = []
    quote = []
    suggestion = []
        
    all_regex.append("1+4*1+1+4*1+1+4*1+") # append all patterns
    character.append("Lack of Confidence") #corresponding character
    quote.append("Nothing builds confidence quicker than practice interviewing. If you don't succeed but atleast try, experience works a lot.y: Do not foster the fear of not getting this particular job.") #corresponding quote
    suggestion.append("Watch this youtube link https://youtu.be/M4iHa7gHvNI") #Do this and watch this
    
    all_regex.append("1*2+4*1+4*1*2+1*") # append all patterns
    character.append("Lack of Preparation") #corresponding character
    quote.append("A lack of preparation gives you a major disadvantage over other candidates and will cost you a job before you have even walked in the door.") #corresponding quote
    suggestion.append("Read this website https://isarta.com/news/job-interview-lack-of-preparation-is-the-biggest-cause-of-awkwardness/") #Do this and watch this
    
    all_regex.append("0*3+4*3+4*0*3+0*") # append all patterns
    character.append("Lack of communication between interviewer") #corresponding character
    quote.append(" Be more relaxed and try to build good relationship between interviewer.") #corresponding quote
    suggestion.append("Read this blog https://www.flexjobs.com/employer-blog/interview-communication-failures-fix/ ") #Do this and watch this
    
    all_regex.append("5*6+4*6+4*5+6+5*") # append all patterns
    character.append("Over Enthusiastic") #corresponding character
    quote.append("Excitement at certain times comes off as insincere and it can make you babble, saying a lot of words that don't mean anything. Focus on answering rather than talking too much.") #corresponding quote
    suggestion.append("Read this blog https://www.chartercollege.edu/news-hub/can-you-be-too-enthusiastic-job-interview ") #Do this and watch this
    
    all_regex.append("^0*1+3*2+3*1*2*") # append all patterns
    character.append("Stage Fear") #corresponding character
    quote.append("Stage fright is very common and could be overcomed through step by step processes, but stuttering is a fright that takes time to conquer.") #corresponding quote
    suggestion.append("Watch this youtube link https://youtu.be/_Tpw8_B-2io ") #Do this and watch this
    
    
    interval_range=[]
    
    for i in range(len(all_regex)):
        match = re.findall(all_regex[i], string)
        #print(match)

        first=0
        last=len(string)-1
        for matches in match:
            
            interval={}
            res = re.search(matches, string[first:last])
            interval['T1']=array_frame[(res.start()+first+1)*frame_diff] #T1 included
            interval['T2']=array_frame[(res.end()+first)*frame_diff] #T2 included
            interval['character']=character[i]
            interval['quote']=quote[i]
            interval['suggestion']=suggestion[i]
            interval_range.append(interval)
            first=res.end()

            if(MinThershold(interval['T1'],interval['T2'])==False):#Checking for Pattern Lenght Thershold
                interval_range.pop()
            
            
        
    return interval_range



def background_substraction(prev,current):
  return np.sum(np.sum(np.absolute(current-prev)))/(np.shape(current)[0]*np.shape(current)[0]);


def emotionModule(vid,vid_path):
    each_frame_emotion=pd.DataFrame(columns=["Participant","Frame No","Emotion"])
    cap=cv2.VideoCapture(vid_path);
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    
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
          prev = current+60

        
        if (abs(background_substraction(prev,current))>60):
          try:
              gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
              cascadePath = "haarcascades\haarcascade_frontalface_default.xml"
              faceCascade = cv2.CascadeClassifier(cascadePath);
              obj = DeepFace.analyze(img, actions = ['emotion'])
              #objs = DeepFace.analyze(["img1.jpg", "img2.jpg", "img3.jpg"]) #analyzing multiple faces same time
              #toc=time.time()
              #print("--------\n")
              #print("Time is "+str((toc-tic)//(3600))+" hours")
              print(" [+ % of processing ]"+str(round(100*count/frame_count,2)))
              #print("--------\n")
              
              value_emotion[obj["dominant_emotion"].lower()]+=1
              
              each_frame=[]
              each_frame=[vid[0:len(vid)-4]]
              each_frame.append(count)
              each_frame.append(obj["dominant_emotion"])
              
              each_frame=np.array(each_frame)
              each_frame=np.reshape(each_frame,(each_frame.shape[0],1))  
              #print(np.shape(each_frame))              
              
              each_frame_emotion_df2=pd.DataFrame(each_frame.transpose(),columns=["Participant","Frame No","Emotion"])
              #temp=each_frame_emotion_df2.iat[0,2]
              #print(temp)
              
              each_frame_emotion=pd.concat([each_frame_emotion,each_frame_emotion_df2])
              #print(each_frame_emotion)
              each_frame_emotion.to_excel("Each Frame Emotion.xlsx",sheet_name='Sheet_name_1')
              prev=current
          except:
              pass
        
    
    total=0
    
    entry_list=[]
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
    
    
    df2=pd.DataFrame(entry_list.transpose(),columns=["angry","fear","neutral","sad","disguist","happy","surprise"])
    return each_frame_emotion,df2
    


def drawPoints(image, faceLandmarks, startpoint, endpoint, isClosed=False):
  points = []
  for i in range(startpoint, endpoint+1):
    point = [faceLandmarks.part(i).x, faceLandmarks.part(i).y]
    points.append(point)

  points = np.array(points, dtype=np.int32)
  cv2.polylines(image, [points], isClosed, (255, 200, 0), thickness=2, lineType=cv2.LINE_8)

# Use this function for 70-points facial landmark detector model
# we are checking if points are exactly equal to 68, then we draw all those points on face one by one
def facePoints(image, faceLandmarks):
    assert(faceLandmarks.num_parts == 68)
    drawPoints(image, faceLandmarks, 0, 16)           # Jaw line
    drawPoints(image, faceLandmarks, 17, 21)          # Left eyebrow
    drawPoints(image, faceLandmarks, 22, 26)          # Right eyebrow
    drawPoints(image, faceLandmarks, 27, 30)          # Nose bridge
    drawPoints(image, faceLandmarks, 30, 35, True)    # Lower nose
    drawPoints(image, faceLandmarks, 36, 41, True)    # Left eye
    drawPoints(image, faceLandmarks, 42, 47, True)    # Right Eye
    drawPoints(image, faceLandmarks, 48, 59, True)    # Outer lip
    drawPoints(image, faceLandmarks, 60, 67, True)    # Inner lip

# Use this function for any model other than
# 70 points facial_landmark detector model
#def facePoints2(image, faceLandmarks, color=(0, 255, 0), radius=4):
#  for p in faceLandmarks.parts():
#    cv2.circle(im, (p.x, p.y), radius, color, -1)

def getLandmarksFromTXTFile(file):
    with open(file) as fp:
        landmarks = []
        for i, line in enumerate(fp):
            landmarks.append([float(k) for k in line.split(' ')])
        landmarks = np.array(landmarks).reshape((-1,136))
       # landmarks = np.resize(landmarks, (136,))
    res = []
    for i in range(136):
      res.append(landmarks[0][i])
    return res

def writeFaceLandmarksToLocalFile(faceLandmarks, fileName):
  with open("./pts",'w') as f:
    for p in faceLandmarks.parts():
      f.write("%f %f\n" %(int(p.x),int(p.y)))
  f.close()


def DetectFaceLandmark(vid,vid_path):
    df=pd.DataFrame(columns=["Participant","Frame No","p1_x","p1_y","p2_x","p2_y","p3_x","p3_y","p4_x","p4_y","p5_x","p5_y","p6_x","p6_y","p7_x","p7_y","p8_x","p8_y","p9_x","p9_y","p10_x","p10_y","p11_x","p11_y","p12_x","p12_y","p13_x","p13_y","p14_x","p14_y","p15_x","p15_y","p16_x","p16_y","p17_x","p17_y","p18_x","p18_y","p19_x","p19_y","p20_x","p20_y","p21_x","p21_y","p22_x","p22_y","p23_x","p23_y","p24_x","p24_y","p25_x","p25_y","p26_x","p26_y","p27_x","p27_y","p28_x","p28_y","p29_x","p29_y","p30_x","p30_y","P31_x","p31_y","P32_x","p32_y","P33_x","p33_y","P34_x","p34_y","P35_x","p35_y","P36_x","p36_y","P37_x","p37_y","P38_x","p38_y","P39_x","p39_y","P40_x","p40_y","P41_x","p41_y","P42_x","p42_y","P43_x","p43_y","P44_x","p44_y","P45_x","p45_y","P46_x","p46_y","P47_x","p47_y","P48_x","p48_y","P49_x","p49_y","P50_x","p50_y","P51_x","p51_y","P52_x","p52_y","P53_x","p53_y","P54_x","p54_y","P55_x","p55_y","P56_x","p56_y","P57_x","p57_y","P58_x","p58_y","P59_x","p59_y","P60_x","P60_y","P61_x","p61_y","P62_x","p62_y","P63_x","p63_y","P64_x","p64_y","P65_x","P65_y","P66_x","p66_y","P67_x","p67_y","P68_x","p68_y"])
    Model_PATH = 'shape_predictor_68_face_landmarks.dat'
    
    # now from the dlib we are extracting the method get_frontal_face_detector()
    # and assign that object result to frontalFaceDetector to detect face from the image with 
    # the help of the 68_face_landmarks.dat model
    frontalFaceDetector = dlib.get_frontal_face_detector()

    # Now the dlip shape_predictor class will take model and with the help of that, it will show 
    faceLandmarkDetector = dlib.shape_predictor(Model_PATH)
    
    df=pd.DataFrame(columns=["Frame No","p1_x","p1_y","p2_x","p2_y","p3_x","p3_y","p4_x","p4_y","p5_x","p5_y","p6_x","p6_y","p7_x","p7_y","p8_x","p8_y","p9_x","p9_y","p10_x","p10_y","p11_x","p11_y","p12_x","p12_y","p13_x","p13_y","p14_x","p14_y","p15_x","p15_y","p16_x","p16_y","p17_x","p17_y","p18_x","p18_y","p19_x","p19_y","p20_x","p20_y","p21_x","p21_y","p22_x","p22_y","p23_x","p23_y","p24_x","p24_y","p25_x","p25_y","p26_x","p26_y","p27_x","p27_y","p28_x","p28_y","p29_x","p29_y","p30_x","p30_y","P31_x","p31_y","P32_x","p32_y","P33_x","p33_y","P34_x","p34_y","P35_x","p35_y","P36_x","p36_y","P37_x","p37_y","P38_x","p38_y","P39_x","p39_y","P40_x","p40_y","P41_x","p41_y","P42_x","p42_y","P43_x","p43_y","P44_x","p44_y","P45_x","p45_y","P46_x","p46_y","P47_x","p47_y","P48_x","p48_y","P49_x","p49_y","P50_x","p50_y","P51_x","p51_y","P52_x","p52_y","P53_x","p53_y","P54_x","p54_y","P55_x","p55_y","P56_x","p56_y","P57_x","p57_y","P58_x","p58_y","P59_x","p59_y","P60_x","P60_y","P61_x","p61_y","P62_x","p62_y","P63_x","p63_y","P64_x","p64_y","P65_x","P65_y","P66_x","p66_y","P67_x","p67_y","P68_x","p68_y"])
    print(f'[+] Processing : {vid_path}')
    vid=cv2.VideoCapture(vid_path)
    frame_count = vid.get(cv2.CAP_PROP_FRAME_COUNT)
    count = 0
    while count<=frame_count-1:
        count =count+1
        ret, img=vid.read()
        imageRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        allFaces = frontalFaceDetector(imageRGB, 0)

        for k in range(0,len(allFaces)):
          faceRectangleDlib = dlib.rectangle(int(allFaces[k].left()),int(allFaces[k].top()),
          int(allFaces[k].right()),int(allFaces[k].bottom()))

      # Now we are running loop on every detected face and putting landmark on that with the help of faceLandmarkDetector
          detectedLandmarks = faceLandmarkDetector(imageRGB, faceRectangleDlib)
      
      # Svaing the landmark one by one to the output folder
          each_frame=[]
          each_frame.append(count)
          for coordinate in detectedLandmarks.parts():
              x=coordinate.x
              y=coordinate.y
              each_frame.append(x)
              each_frame.append(y)
      # Now finally we drawing landmarks on face
          df = pd.DataFrame([each_frame],columns = ["Frame No","p1_x","p1_y","p2_x","p2_y","p3_x","p3_y","p4_x","p4_y","p5_x","p5_y","p6_x","p6_y","p7_x","p7_y","p8_x","p8_y","p9_x","p9_y","p10_x","p10_y","p11_x","p11_y","p12_x","p12_y","p13_x","p13_y","p14_x","p14_y","p15_x","p15_y","p16_x","p16_y","p17_x","p17_y","p18_x","p18_y","p19_x","p19_y","p20_x","p20_y","p21_x","p21_y","p22_x","p22_y","p23_x","p23_y","p24_x","p24_y","p25_x","p25_y","p26_x","p26_y","p27_x","p27_y","p28_x","p28_y","p29_x","p29_y","p30_x","p30_y","P31_x","p31_y","P32_x","p32_y","P33_x","p33_y","P34_x","p34_y","P35_x","p35_y","P36_x","p36_y","P37_x","p37_y","P38_x","p38_y","P39_x","p39_y","P40_x","p40_y","P41_x","p41_y","P42_x","p42_y","P43_x","p43_y","P44_x","p44_y","P45_x","p45_y","P46_x","p46_y","P47_x","p47_y","P48_x","p48_y","P49_x","p49_y","P50_x","p50_y","P51_x","p51_y","P52_x","p52_y","P53_x","p53_y","P54_x","p54_y","P55_x","p55_y","P56_x","p56_y","P57_x","p57_y","P58_x","p58_y","P59_x","p59_y","P60_x","P60_y","P61_x","p61_y","P62_x","p62_y","P63_x","p63_y","P64_x","p64_y","P65_x","P65_y","P66_x","p66_y","P67_x","p67_y","P68_x","p68_y"]).append(df)
          print(" [+ % of processing ]"+str(round(100*count/frame_count,2)))
    return df
    


def headGesture(faceLandmarkspd):
    
    head_tracking = faceLandmarkspd.loc[:,'p28_y'].to_numpy()
    
    stableCount=0
    transientCount=0
    extremeCount=0
    data=[]
    
    for index in range(len(head_tracking)):
        if(index > 2 and index < len(head_tracking)):
            if(max(head_tracking[index-2:index+3]) - min(head_tracking[index-2:index+3]) <= 2):
                stableCount=stableCount+1
            elif((max(head_tracking[index-2:index+3])== head_tracking[index]) or (min(head_tracking[index-2:index+3]) == head_tracking[index])):
                extremeCount=extremeCount+1
            else:
                transientCount=transientCount+1
                
    data.append([100*stableCount/(stableCount+transientCount+extremeCount),100*extremeCount/(stableCount+transientCount+extremeCount)])
    head_model=joblib.load( r"C:/Users/muham/Desktop/Mlabs/Final DataSet/random_forest.joblib")
    #print(data)
    return round(head_model.predict(data)[0],2)
    

    
each_frame_pd,average_pd=emotionModule("P5.avi",r"C:\Users\muham\Desktop\Mlabs\Data\Videos\P5.avi")
faceLandmarkspd=DetectFaceLandmark("P5.avi",r"C:\Users\muham\Desktop\Mlabs\Data\Videos\P5.avi")
#print(each_frame_pd)
#print(average_pd)



#average_pd = pd.read_csv(r"C:\Users\muham\Desktop\AverageOfP44.csv")
print(globalScore(average_pd,emotion_thershold))


#each_frame_pd = pd.read_csv(r"C:\Users\muham\Desktop\EachFrameOfP44.csv") 
print(TimeIntervalScoreSad(each_frame_pd,emotion_thershold,interval_value=75))
print(TimeIntervalScoreFear(each_frame_pd,emotion_thershold,interval_value=75))
print(TimeIntervalScoreAngry(each_frame_pd,emotion_thershold,interval_value=75))
print(FindPattern(each_frame_pd,emotion_number))


print("Score is "+str(headGesture(faceLandmarkspd))+" out of 7")