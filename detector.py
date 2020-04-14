import sys
import numpy as np
import pandas as pd
import math
from matplotlib import pyplot as plt
import cv2
import csv
from numpy import genfromtxt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import os

def main(image_name):
    car = cv2.imread(image_name,0)
    car = cv2.medianBlur(car,7)
    dst = cv2.cornerHarris(car,2,3,0.04)
    dst = cv2.dilate(dst,None)

    for i in range(dst.shape[0]):
        for j in range(dst.shape[1]):
            if dst[i][j] < 0:
                dst[i][j] = 0
                
    norm_dst = cv2.normalize(dst, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    norm_dst.astype(np.float32)

    thres = np.amax(norm_dst) * 0.05

    for i in range(norm_dst.shape[0]):
        for j in range(norm_dst.shape[1]):
            if norm_dst[i][j] > thres:
                norm_dst[i][j] = 255
            else:
                norm_dst[i][j] = 0

    ret,thresh = cv2.threshold(norm_dst,0,255,0)
    thresh = np.array(thresh , dtype = np.uint8)
    connectivity = 4 
    output = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_32S)


    # Get the results
    # The first cell is the number of labels
    num_labels = output[0]
    # The second cell is the label matrix
    labels = output[1]
    labels = np.uint8(labels)
    # The third cell is the stat matrix
    stats = output[2]
    # The fourth cell is the centroid matrix
    centroids = output[3]

    labels_max = np.amax(labels)
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            if (labels[i][j] > labels_max * 0.3):
                labels[i][j] = 255
            else:
                labels[i][j] = 0

    ret,thresh = cv2.threshold(labels,0,255,0)
    thresh = np.array(thresh , dtype = np.uint8)
    _,contours, hierarchy=cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    # Concatenate all contours
    cnts = np.concatenate(contours)
    # Determine and draw bounding rectangle
    x , y, w , h = cv2.boundingRect(cnts)
    # x = x - 20
    w = w + 15

    letter = car[y:y+h,x:x+w]
    cv2.imwrite("output.png", letter)
        

if __name__ == "__main__":
    main(sys.argv[1])
