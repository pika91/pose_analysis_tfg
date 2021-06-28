import cv2
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
    entry = 'salida_dani'
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


def randrange(n, vmin, vmax):
    '''
    Helper function to make an array of random numbers having shape (n, )
    with each number distributed Uniform(vmin, vmax).
    '''
    return (vmax - vmin)*np.random.rand(n) + vmin

def readCSV(nameFile):    
    dfCompare = pd.read_csv(nameFile, header=None)
    dfCompare = dfCompare[1:].values.tolist()
    for i in range(len(dfCompare)):
        del dfCompare[i][0]
    return dfCompare

def readVideoFrames(entry,entry2):    
    #Lectura de los frames del video, para poder comparar frames en la fase de DTW
    vidcap = cv2.VideoCapture("./media/"+entry+'.mp4')
    success,image = vidcap.read()
    frames_1 = []
    count = 0
    while success:
      frames_1.append(image)    
      success,image = vidcap.read()
    
    vidcap = cv2.VideoCapture("./media/"+entry2+'.mp4')
    success,image = vidcap.read()
    frames_2 = []
    count = 0
    while success:
      frames_2.append(image)    
      success,image = vidcap.read()
      
    return frames_1,frames_2

def drawPersonLines(keypointsPositions,keypointsValue,ax):
    for i in keypointsPositions:
        keyIndex = keypointsPositions[i]
        x = [keypointsValue[0][keyIndex[0]],keypointsValue[0][keyIndex[1]]]
        y = [keypointsValue[2][keyIndex[0]],keypointsValue[2][keyIndex[1]]]
        z = [keypointsValue[1][keyIndex[0]],keypointsValue[1][keyIndex[1]]]
        if 0 not in x and 0 not in z:  
            ax.plot(x, y, z, c = 'b')

    return False

keypointsPositions = {'cuello': [0,1], 'hombro_izq':[1,2],'hombro_der':[1,5],'hombro_codo_izq':[2,3],'hombro_codo_der':[5,6],'codo_mano_izq':[3,4],'codo_mano_der':[6,7]
                      ,'cuerpo':[1,8],'cadera_izq':[8,9],'cadera_der':[8,12],'pierna_sup_izq':[9,10],'pierna_sup_der':[12,13],'pierna_inf_izq':[10,11],'pierna_inf_der':[13,14],
                      'cuello_ojo_izq':[0,15],'ojo_izq':[15,17],'cuello_ojo_der':[0,16],'ojo_der':[16,18],'talon_izq':[11,24],'talon_der':[14,21],'pie_izq':[11,22],'pie_der':[14,19],
                      'pie_inf_izq':[22,23],'pie_inf_der':[19,20],}

entry = 'salida_dani'
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

    
#Analisis 
#Forzamos el 1 ya que conocemos que es el saltador
#Todo escoger personas con imagenes
    
persona = person_detection[1]
data = []
#Calcular angulo pierna derecha
entry = 'salida_dani_analisi'
video_analisis = Video(input_path="./media/"+entry+".mp4")
for i,frame_analisis in enumerate(video_analisis):
    if i in persona.keypoints:
        data.append(persona.keypoints[i])
    else:
        zero_array = np.zeros((25,2))
        data.append(zero_array)
    
    video_analisis.write(frame_analisis)
        
arr = np.array(data[::])


keypointsInFrame = []
for keypointId in range(arr.shape[1]):
    xArray = []
    yArray = []

    for frameNum in range(0,arr.shape[0],25):
        xArray.append(arr[:,:][frameNum][keypointId,0])
        yArray.append(arr[:,:][frameNum][keypointId,1])
        
    keypointsInFrame.append([xArray,yArray]);
    
framesKeypoints = []
fastDTWPoints = []
for frameNum in range(0,arr.shape[0],5):
    xArray = []
    yArray = []
    frameArrayNum = []
    xyArray = []
    keypointsGroup = arr[:,:][frameNum]
    for kpg in keypointsGroup:
        xArray.append(kpg[0])
        yArray.append(kpg[1])
        frameArrayNum.append(frameNum)
        xyArray.append(kpg[0])
        xyArray.append(kpg[1])

    fastDTWPoints.append(xyArray)
    framesKeypoints.append([xArray,yArray,frameArrayNum]);
        
"""fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for keypointsValue in keypointsInFrame:
    ax.scatter(keypointsValue[0],range(len(keypointsValue[0])),keypointsValue[1])

ax.set_xlabel('X Label')
ax.set_zlabel('Y Label')
ax.set_ylabel('Frame')
ax.set_xlim(2000,0)
#ax.set_ylim(20,160)
ax.set_zlim(1000,0)


plt.show()"""


        
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for keypointsValue in framesKeypoints:
    ax.scatter(keypointsValue[0],keypointsValue[2],keypointsValue[1])
    drawPersonLines(keypointsPositions,keypointsValue,ax)

ax.set_xlabel('X Label')
ax.set_zlabel('Y Label')
ax.set_ylabel('Frame')
ax.set_xlim(1600,100)
ax.set_zlim(1600,100)

plt.show()


#Save Data
arr = np.array(fastDTWPoints)
df = pd.DataFrame(data=arr)
print("---- SAVE DATA IN CSV --- ")
df.to_csv(entry+'_allPoints_data.csv',index=False);
print("---- END DATA IN CSV --- ")


fastX = fastDTWPoints
fastY = readCSV('salida_paula_analisi_allPoints_data.csv')
distance, path = fastdtw(fastX, fastY, dist=euclidean)



frames1, frames2 = readVideoFrames('salida_dani','salida_paula')
videoComparativa = []
i = 0
for xy in path:
    img1 = frames1[xy[0]*5]
    img2 = frames2[xy[1]*5]
    vis = np.concatenate((img1, img2), axis=1)
    cv2.imwrite('./output/comparativa/out'+str(i)+'.png', vis)
    i+=1
    videoComparativa.append(vis)

out = cv2.VideoWriter('comparativa.mp4',cv2.VideoWriter_fourcc(*'H264'), 30, (videoComparativa[0].shape[1],videoComparativa[0].shape[0]))
for i in range(len(videoComparativa)):
    out.write(videoComparativa[i])
out.release()


res = dtw(np.asarray(fastX),np.asarray(fastY),step_pattern="asymmetric", open_begin=True, open_end=True)
res.plot_path()

videoComparativa = []
i = 0
for xy in res.path:
    img1 = frames1[xy[0]*5]
    cv2.putText(img1,'Frame:'+str(xy[0]*5),(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255))
    img2 = frames2[xy[1]*5]
    cv2.putText(img2,'Frame:'+str(xy[1]*5),(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255))
    vis = np.concatenate((img1, img2), axis=1)
    cv2.imwrite('./output/comparativaDTW/out'+str(i)+'.png', vis)
    i+=1
    videoComparativa.append(vis)

out = cv2.VideoWriter('comparativadtw.mp4',cv2.VideoWriter_fourcc(*'H264'), 5, (videoComparativa[0].shape[1],videoComparativa[0].shape[0]))
for i in range(len(videoComparativa)):
    out.write(videoComparativa[i])
out.release()



