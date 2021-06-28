import cv2
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import math


import norfair
from norfair import Detection, Tracker, Video

from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from dtwalign import dtw


class Person:
  def __init__(self, id_person , start_frame):
    self.keypoints = dict()
    self.start_frame = start_frame
    self.id_person = id_person
    self.frame_box = []
    self.has_frame = False

class DetectionFrame(Detection):
    def __init__(self, points, frameId):
        super().__init__(points)
        self.frameId = frameId


# Distance function
def centroid_distance(detection, tracked_object):
    #dist = np.linalg.norm(detection.points - tracked_object.estimate)
    dist = np.linalg.norm(detection.points[8] - tracked_object.estimate[8])
    return dist

def getOpenPoseKeypoints(frame_idx,only_one):
    entry = 'long_jump_01'
    entries_dir = os.listdir('output/'+entry)
    df_json = pd.read_json('output/'+entry+'/'+entries_dir[frame_idx])
    poses = []
    for person in df_json['people']:
        keypoints=[]
        for x in range(0,len(person['pose_keypoints_2d']),3):
            keypoints.append(person['pose_keypoints_2d'][x:x+3])
        poses.append(keypoints)
        
    #Eliminamos confianza y dejamos solo valores x e y de los keypoints
    poses = np.array(poses)
    if poses.shape[0]>0:
        poses = poses[:,:,:2]
        
        if only_one == True:
            return poses[:,8]
        else:
            return poses
    else:
        return poses

def poses2boxes(poses):
    boxes = []
    for person in poses:
        seen_bodyparts = person[np.where((person[:,0] != 0) | (person[:,1] != 0))]

        x1 = min(seen_bodyparts[:,0])
        x2= max(seen_bodyparts[:,0])
        y1 = min(seen_bodyparts[:,1])
        y2= max(seen_bodyparts[:,1])

        deviation = 10
        box = [int(x1-deviation), int(y1-deviation), int(x2+deviation), int(y2+deviation)]
        boxes.append(box)
    return np.array(boxes)

#TODO Person crop frame
def person2boxes(poses):
    box = []
    return np.array(box)

#Función para calcular el angulo entre dos vectores
def calcularAngulo(p1,p2,p3):
    angle = math.degrees(math.atan2(p1[1] - p2[1],p1[0] - p2[0] ) -math.atan2(p3[1] - p2[1],p3[0] - p2[0] ))
    if angle < 0 :
        angle = int(angle + 360)
    return angle

def printVectorInImg(img,p1,p2,p3):
    angulo = calcularAngulo(p1,p2,p3)
    cv2.putText(img, 'Angulo: '+str(int(angulo)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
    cv2.line(img,(int(p1[0]),int(p1[1])),(int(p2[0]),int(p2[1])),(255,255,255),3)
    cv2.line(img,(int(p3[0]),int(p3[1])),(int(p2[0]),int(p2[1])),(255,255,255),3)


entry = 'long_jump_01'
# Norfair
video = Video(input_path="./media/"+entry+".mp4")
tracker = Tracker(distance_function=centroid_distance, distance_threshold=120)
count = 0
person_detection = dict()
output_frame = []
for i,frame in enumerate(video):
    poses = getOpenPoseKeypoints(i,False)
        
    pose_detection = getOpenPoseKeypoints(i,True)
    detections =[ DetectionFrame(points=pose,frameId=i) for pose in poses]
    tracked_objects = tracker.update(detections=detections)
    posesDraw = []
    for tr in tracked_objects:
        if tr.id not in person_detection:
            person_detection[tr.id] = Person(tr.id,i-1)
        person_detection[tr.id].keypoints[tr.last_detection.frameId] = tr.last_detection.points


print("---- BOXES --- ")
video = Video(input_path="./media/"+entry+".mp4")
for i,out_frame in enumerate(video):
    posesDraw = []
    personId = []
    for p in person_detection:
        person = person_detection[p]
        if i in person.keypoints:
            posesDraw.append(person.keypoints[i]);
            personId.append(person.id_person)
    posesDraw = np.array(posesDraw)
    boxes = poses2boxes(posesDraw)
    for b, box in enumerate(boxes):
        personFrame = out_frame[box[1]:box[3],box[0]:box[2]]
        person = person_detection[personId[b]]
        if person.has_frame == False:
            person.frame_box = personFrame
            person.has_frame = True
        cv2.rectangle(out_frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),(255,255,0), 2)
    
    video.write(out_frame)
print("---- END BOXES --- ")

"""print("---- SHOW PERSON BOX --- ")
for p in person_detection:
        person = person_detection[p]
        if person.has_frame == True:
            cv2.imshow('Detected Person',person.frame_box)
            cv2.waitKey(0)
print("---- END SHOW PERSON BOX --- ")"""


#HAR Filter
    
#Analisis 
#Forzamos el 1 ya que conocemos que es el saltador
#Todo escoger personas con imagenes
    
persona = person_detection[1]
data = []
#Calcular angulo pierna derecha
entry = 'long_jump_01_analisi'
video_analisis = Video(input_path="./media/"+entry+".mp4")
for i,frame_analisis in enumerate(video_analisis):
    if i in persona.keypoints:
        cadera = persona.keypoints[i][12]
        rodilla = persona.keypoints[i][13]
        tobillo = persona.keypoints[i][14]
        angle = calcularAngulo(cadera,rodilla,tobillo)
        printVectorInImg(frame_analisis,cadera,rodilla,tobillo)
        data.append(angle)
    else:
        data.append(0)
    
    video_analisis.write(frame_analisis)
        
arr = np.array(data[::])

df = pd.DataFrame(data=arr.flatten())
maData = df.rolling(window=5).mean()
"""
GRAFICO ANGULO CON SMOOTH
plt.plot(data[::], label='Atleta profesional 1')
plt.plot(maData, label='Atleta profesional 2')

plt.ylabel('Ángulo')
plt.xlabel('Frame')
plt.title(" Ángulo pierna izquierda Salto")
plt.legend()
plt.show()
"""


print("---- SAVE DATA IN CSV --- ")
maData.to_csv(entry+'_data.csv');
print("---- END DATA IN CSV --- ")

print("---- COMPARE DTW--- ")

dfCompare = pd.read_csv('long_jump_02_analisi_data.csv', header=None)

"""plt.plot(dfCompare[1], label="Atleta profesional 2")
plt.plot(maData, label="Atleta profesional 1")

plt.ylabel('Ángulo')
plt.xlabel('Frame')
plt.title(" Ángulo pierna izquierda Salto")
plt.legend()
plt.show()"""

fastX = dfCompare[1][:].to_numpy()
fastY = maData[0][:].to_numpy()

fastX = fastX[5:]
fastY = fastY[5:]
distance, path = fastdtw(fastX, fastY, dist=euclidean)

res = dtw(fastX,fastY,step_pattern="asymmetric", open_begin=True, open_end=True)
#res.plot_path()

print(distance)
print(path)

plt.plot(fastX, label="Atleta profesional 2")
plt.plot(fastY, label="Atleta profesional 1")
plt.title(" DTW Ángulo pierna izquierda Salto")
for x in range(0,len(res.path),10):
    x_values = [res.path[x][0],res.path[x][1]]
    print(x_values)
    #print(x_values)
    y_values = [fastX[res.path[x][0]],fastY[res.path[x][1]]]
    plt.plot(x_values, y_values,linewidth=0.5, color='black')

plt.ylabel('Ángulo')
plt.xlabel('Frame')
plt.legend()
plt.show()

