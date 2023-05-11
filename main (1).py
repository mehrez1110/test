#!/usr/bin/env python
# coding: utf-8

# In[2]:


import json

import cv2 as cv
import numpy as np
import skimage.io as io
import xmltodict
import matplotlib.pyplot as plt
from multiprocessing import Process
import time

# scale

# from Rect import *


# Utility functions for image preprocessing

# For variance normalization
# we need:
# 1- integral of the image
# 2- integral of the square of the image
# 3- calulate the the varia
#
# var=mean^2 -1/N(sum(X^2))

# In[3]:


# var=mean^2 -1/N(sum(X^2))
def calculateVariance(window, sqrdWindow):
    # Calculate mean value of the image and square it
    # sqrdMean=(np.mean(window))**2
    mean = ((window[1][1] + window[-1][-1] - window[1][-1] - window[-1][1]) / np.size(window))

    # Sum of pixels in integral of the squared image
    # sqrdSum=np.sum(sqrdWindow)
    sqrdSum = (sqrdWindow[1][1] + sqrdWindow[-1][-1] - sqrdWindow[1][-1] - sqrdWindow[-1][1])

    # Number of pixels in the window
    N = np.size(window)

    # Calculated variance
    variance = (sqrdSum / N) - (mean * mean)

    # Normalize the window and return it
    # normalizedWindow=window*variance

    return variance


# In[4]:


# calculate integral of the image(to be updated for better optimization)
# def integrateImage(img,i,j,cum):
def integrateImage(img):
    rows = img.shape[0]
    cols = img.shape[1]
    integralImage = np.zeros((rows + 1, cols + 1))
    integralImage[1:, 1:] = np.cumsum(img, 1)
    integralImage = np.cumsum(integralImage, 0)

    # return outputImage[i][j]
    return integralImage


# Show the figures / plots inside the notebook
def show_images(images,titles=None):
    #This function is used to show image(s) with titles by sending an array of images and an array of associated titles.
    # images[0] will be drawn with the title titles[0] if exists
    # You aren't required to understand this function, use it as-is.
    n_ims = len(images)
    if titles is None: titles = ['(%d)' % i for i in range(1,n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image,title in zip(images,titles):
        a = fig.add_subplot(1,n_ims,n)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show()




class Rect:
    def __init__(self, inputText, y1=0, x1=0, width=0, height=0, weight=0.0, x2=0, y2=0):
        splitInput = inputText.split(" ")

        self.x1 = int(splitInput[0]) - 1
        self.y1 = int(splitInput[1]) - 1

        self.width = int(splitInput[2])
        self.height = int(splitInput[3])

        self.weight = float(splitInput[4])

        self.y2 = self.y1 + self.height
        self.x2 = self.x1 + self.width

    def calculateArea(self, window, scale):
        # self.printRect()
        # print("window values: ",window[self.y2][self.x2],window[self.y1][self.x1],window[self.y1][self.x2],window[self.y2][self.x1])
        scaledX1 = self.x1 * scale
        scaledY1 = self.y1 * scale
        scaledY2 = self.y2 * scale
        scaledX2 = self.x2 * scale
        area = (window[scaledY2][scaledX2] + window[scaledY1][scaledX1] - window[scaledY1][scaledX2] - window[scaledY2][
            scaledX1])
        # print("weight*area/size: ",(self.weight*area)/(np.size(window)))
        # return (self.weight*area)/((window[0][0]+window[-1][-1]-window[0][-1]-window[-1][-0]))
        return (self.weight * area) / np.size(window)

    def printRect(self):
        print('Rect')
        print('x1: ', self.x1)
        print('x2: ', self.x2)
        print('y1: ', self.y1)
        print('y2: ', self.y2)
        print('width: ', self.width)
        print('height: ', self.height)
        print('weight: ', self.weight)


class Feature:
    def __init__(self, leftValue, rightValue, threshold, rects):
        self.leftValue = leftValue
        self.rightValue = rightValue
        self.threshold = threshold
        self.rects = rects

    # Calculate initial value of the feature(two or three rectangle value)
    def calculateValue(self, window, scale):
        initialValue = 0
        for i in range(0, len(self.rects)):
            initialValue += self.rects[i].calculateArea(window, scale)
        return initialValue

    # weak classifier
    def classify(self, window, variance, scale):

        value = self.calculateValue(window, scale)
        if (value > self.threshold * variance):
            # print('feature right value after weak classifier:',self.rightValue)
            return self.rightValue
        else:
            # print('feature left value after weak classifier:',self.leftValue)
            return self.leftValue

    def printFeature(self):
        print('Feature: ')
        print('Left value: ', self.leftValue)
        print('Right value: ', self.rightValue)
        print('Threshold: ', self.threshold)
        for i in range(0, len(self.rects)):
            self.rects[i].printRect()


# Stage class

# In[10]:


# Stages of the cascaded classifier
# We have 25 stages numbered from 0 to 24
class Stage:
    def __init__(self, features, threshold):
        self.features = features
        self.threshold = threshold

    # Validate if a window should pass a stage
    def validateWindow(self, window, variance, scale):
        sum = 0
        for i in range(0, len(self.features)):
            sum += self.features[i].classify(window, variance, scale)
        if (sum > self.threshold):
            return True
        else:
            return False

    def printStage(self):
        print('Stage threshold', self.threshold)
        for i in range(0, len(self.features)):
            self.features[i].printFeature()

        # Xml to json converter


# In[11]:


# Read haar cascade xml file and convert to json
def xmlToJson(xmlFile):
    with open(xmlFile) as xml_file:
        data_dict = xmltodict.parse(xml_file.read())
    json_data = json.loads(json.dumps(data_dict))
    return json_data


# In[12]:


# Call xmlToJson function
json_data = xmlToJson('haarcascade_frontalface_default.xml')
textRect = json_data['opencv_storage']['haarcascade_frontalface_default']['stages']["_"][0]['trees']['_'][0]['_'][
    'threshold']
print(textRect)

# rect=Rect(textRect)
# print(rect.y1)


# Reading haar cascade frontal face xml file and mapping accordingly

# In[13]:


# Stages allocation

# number of stages
numberOfStages = len(json_data['opencv_storage']['haarcascade_frontalface_default']['stages']['_'])
print(numberOfStages)

# init list of stages
stagesList = []

# list of features per stage
featuresList = []

# unMapped rects list
textRects = []

# list of rects for each feature
rectsList = []

for i in range(0, numberOfStages):
    # allocating stage threshold
    stageThreshold = float(
        json_data['opencv_storage']['haarcascade_frontalface_default']['stages']['_'][i]['stage_threshold'])

    # allocate mumber of features
    numberOfFeatures = len(
        json_data['opencv_storage']['haarcascade_frontalface_default']['stages']['_'][i]['trees']['_'])

    # allocate each feature
    for j in range(0, numberOfFeatures):
        # allocate the feature's left value
        leftValue = float(
            json_data['opencv_storage']['haarcascade_frontalface_default']['stages']['_'][i]['trees']['_'][j]['_'][
                'left_val'])

        # allocate the feature's right value
        rightValue = float(
            json_data['opencv_storage']['haarcascade_frontalface_default']['stages']['_'][i]['trees']['_'][j]['_'][
                'right_val'])

        # allocate the feature's threshold
        featureThreshold = float(
            json_data['opencv_storage']['haarcascade_frontalface_default']['stages']['_'][i]['trees']['_'][j]['_'][
                'threshold'])

        # allocate rects as its xml format
        textRects = \
        json_data['opencv_storage']['haarcascade_frontalface_default']['stages']['_'][i]['trees']['_'][j]['_'][
            'feature']['rects']['_']

        # map rects' xml format to Rect class's format
        for k in range(0, len(textRects)):
            rectsList.append(Rect(textRects[k]))

        # add each feature to features list in the stage
        featuresList.append(Feature(leftValue, rightValue, featureThreshold, rectsList))

        # clear list of rects for each feature
        rectsList = []

        # append stage to the cascade stages list
    stagesList.append(Stage(featuresList, stageThreshold))

    # clear list of features per stage
    featuresList = []


# Cascaded classifier

# In[14]:


def readImage(imageLocation):
    # input image
    finalImage=io.imread(imageLocation,as_gray=False)
    finalImage = cv.resize(finalImage, (384, 288))
    img=io.imread(imageLocation,as_gray=True)
    img = cv.resize(img, (384, 288))
    return img,finalImage
    # show_images([img])

def cascaded_classifier(img,finalImage):
    scale = 1

    # # input image
    # finalImage = imageLocation
    # finalImage = cv.resize(finalImage, (384, 288))
    # img = imageLocation
    # img = cv.resize(img, (384, 288))
    # # show_images([img])

    for l in range(0, 4):
    
        print('scale', scale)

        # calculate integral img
        integralImage = integrateImage(img)

        # calculate integral of the img squared
        sqrdIntegralImage = integrateImage(np.square(img))

        # width and height of the sliding window(24x24)
        windowWidth = int(np.ceil(24 * scale))
        windowHeight = int(np.ceil(24 * scale))

        # windows inits
        currentWindow = np.zeros((windowHeight, windowWidth))
        currentWindowSqrd = np.zeros((windowHeight, windowWidth))
        varNormalizedWindow = np.zeros((windowHeight, windowWidth))

        # boolean to check if a window passes the stage

        # list of passed windows
        passedWindowsList = []
        rectangleIndicesList = []

        # window score variable
        score = 0

        # Main loop(sliding window and detection)
        for i in range(1, integralImage.shape[0] - windowHeight + 1):
            for j in range(1, integralImage.shape[1] - windowWidth + 1):
                # current window of the .integral image
                currentWindow = integralImage[i - 1:i + windowHeight, j - 1:j + windowWidth]

                # current window of integral image squared
                currentWindowSqrd = sqrdIntegralImage[i - 1:i + windowHeight, j - 1:j + windowWidth]
                # variance normalization of the image
                # varNormalizedWindow=varianceNormalize(currentWindow,currentWindowSqrd)
                variance = calculateVariance(currentWindow, currentWindowSqrd)

                # Loop on the cascaded classifiers
                for k in range(0, len(stagesList)):
                    # return a boolean based on the window's validation with a stage
                    passed = stagesList[k].validateWindow(currentWindow, np.sqrt(variance), scale)

                    # stagesList[k].printStage()
                    if (not passed):
                        break
                    else:
                        score += 1

                # print(score)
                if (score == 25):
                    # passedWindowsList.append(img[i:i+windowHeight,j:j+windowWidth])
                    finalImage = cv.rectangle(finalImage, (j, i), (j + windowWidth, i + windowHeight), (0, 0, 255), 1)
                    
                score = 0

        # show_images([finalImage])
        scale = int(np.ceil(scale * (1.25)))
    
    show_images([finalImage])
    


# In[15]:


# imageLoc = io.imread('images/images/1442313353nasa-small.jpg',as_gray=True)
# cascaded_classifier(imageLoc)


# In[ ]:



# In[17]:




if __name__ == "__main__":
    # image = io.imread('1442313353nasa-small.jpg', as_gray=True)
    # imageWidth = image.shape[0]
    # imageHeight = image.shape[1]
    
    image, finalImage = readImage('IMG_0859.jpg')
    imageWidth = image.shape[0]
    imageHeight = image.shape[1]
    
    image2 = image[:imageWidth // 2, :imageHeight // 2]
    image3 = image[imageWidth // 2:, :imageHeight // 2]
    image4 = image[:imageWidth // 2, imageHeight // 2:]
    image5 = image[imageWidth // 2:, imageHeight // 2:]
    
    finalImage2 = image[:imageWidth // 2, :imageHeight // 2]
    finalImage3 = image[imageWidth // 2:, :imageHeight // 2]
    finalImage4 = image[:imageWidth // 2, imageHeight // 2:]
    finalImage5 = image[imageWidth // 2:, imageHeight // 2:]
    
    # new_width = imageWidth//8
    # new_height= imageHeight//8
    # image000=image[:,:new_height]
    # image001=image[:,new_height:new_height*2]
    # image010=image[:,new_height*2:new_height*3]
    # image011=image[:,new_height*3:new_height*4]
    # image100=image[:,new_height*4:new_height*5]
    # image101=image[:,new_height*5:new_height*6]
    # image110=image[:,new_height*6:new_height*7]
    # image111=image[:,new_height*7:]
    
    # finalImage000=image[:,:new_height]
    # finalImage001=image[:,new_height:new_height*2]
    # finalImage010=image[:,new_height*2:new_height*3]
    # finalImage011=image[:,new_height*3:new_height*4]
    # finalImage100=image[:,new_height*4:new_height*5]
    # finalImage101=image[:,new_height*5:new_height*6]
    # finalImage110=image[:,new_height*6:new_height*7]
    # finalImage111=image[:,new_height*7:]
    
    image_list = [image2, image3, image4, image5]
    final_image_list=[finalImage2,finalImage3,finalImage4,finalImage5]
    # image_list=[image000,image001,image010,image011,image100,image101,image110,image111]
    # final_image_list=[finalImage000,finalImage001,finalImage010,finalImage011,finalImage100,finalImage101,finalImage110,finalImage111]
    
    procs = []
    start_time = time.perf_counter()
    for i in range(0,len(image_list)):
        p = Process(target=cascaded_classifier, args=(image_list[i],final_image_list[i]))
        p.start()
        procs.append(p)


    for p in procs:
        p.join()

    finish_time = time.perf_counter()
    
    print(f"Program finished in {finish_time - start_time} seconds")
# In[ ]:




