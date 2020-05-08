#!/usr/bin/env python
# coding: utf-8

# # Module_2_Object_Detection_With_SSD

# # Import Libs

# In[1]:


import numpy as np
import torch
import cv2
import data
import ssd
import imageio

# Importing the libraries
import torch
from torch.autograd import Variable
import cv2
from data import BaseTransform, VOC_CLASSES as labelmap
from ssd import build_ssd
import imageio


# # Define Fxns

# In[2]:


def detect(frame, net, transform):
    height, width = frame.shape[:2] # height width channels
    frame_t = transform(frame)[0]
    
    # convert to torch tensor in GRB order
    x = torch.from_numpy(frame_t).permute(2,0,1) 
    
    # add batch dimension (x.unsqueeze) and define data as variable class for torch
    x = torch.autograd.Variable(x.unsqueeze(0))
    
    # feed x into pre-trained net and get y pred tensor
    y = net(x)
    
    # fetch the detections
    detections = y.data #get data from torch tensor y
    #detections == [batch, num classes, num occurences, (score, x0, y0, x1, y1)]
    
    # define upper left & lower right corner of detected object rectangles
    scale = torch.Tensor([width,height,width,height]) 
    
    # iterate through classes and check whether to keep prediciton or not
    for i in range(detections.size(1)): #i==class
        j = 0 #occurance
        
        #while score of occurance j of class i is larger than 0.6, keep the pred
        while detections[0,i,j,0] > 0.1:
            pt = np.array(detections[0,i,j,1:] * scale) #fetch x0 x1 y0 y1, scale the pts
            cv2.rectangle(frame, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), (255, 0, 0), 2)
            #add text labels to preds on the image
            cv2.putText(frame, labelmap[i-1], (int(pt[0]), int(pt[1])), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2,
                        (255, 255, 255), #text color
                        2, cv2.LINE_AA #display as a line
                       )
            j+=1
            
    return frame


# # Define SSD Neural Net (Pretrained)

# In[3]:


net = ssd.build_ssd('test')

#load the weights
net.load_state_dict(torch.load('ssd300_mAP_77.43_v2.pth',
                               map_location = lambda storage,
                               loc: storage))


# # Transform the image for feed into img

# In[4]:


transform = data.BaseTransform(net.size,
                               (104/256.0, 117/256.0, 123/256.0) #scale the colors
                              )


# # Run Object Detection

# In[5]:


# read the video
reader = imageio.get_reader('epic_horses.mp4') #could also use PIL to read the video

#get frames per second
fps = reader.get_meta_data()['fps']

#create output video
writer = imageio.get_writer('output-horses.mp4',fps = fps)

for i,frame in enumerate(reader): #iterate through the reader video
    #apply detection to the frame
    frame = detect(frame, net.eval(), transform)
    writer.append_data(frame)
    print(i)
writer.close()


# In[11]:


#get_ipython().system("open 'funny_dog_out.mp4'")


# In[ ]:




