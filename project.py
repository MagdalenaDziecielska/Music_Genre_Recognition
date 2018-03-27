# -*- coding: utf-8 -*-

from scipy.io.wavfile import read
import numpy as np
import wave
import struct
import tkinter
from scipy import signal
from scipy.spatial import distance
import librosa as librosa
import librosa.display
import glob
import math
from collections import Counter
import matplotlib.pyplot as plt
from dtw import dtw

windowFFT = 512
windowMFCC = 512

def everyOther (v, offset=0):
   return [v[i] for i in range(offset, len(v), 2)]

def window(windowId, size,beta=1):
    if windowId == 0:
        return np.ones(size) 
    elif windowId == 1:
        return np.hamming(size)
    elif windowId == 2:
        return np.bartlett(size)
    elif windowId == 3:
        return np.blackman(size)
    elif windowId == 4:
        return np.hanning(size)
    elif windowId == 5:
        return np.kaiser(size,beta)


def createSpectrogram(signalTime, windowSize, shift, windowId, beta=1):
    spectrogram = []
    for frame in range(0, len(signalTime) - windowSize, shift):
        fftResult = np.fft.fft(signalTime[frame:frame + windowSize] * window(windowId,windowSize,beta))[0:int(windowSize / 2) + 1]
        convertedFft = pow((pow(fftResult.real, 2) + pow(fftResult.imag, 2)), 0.5)
        spectrogram.append(convertedFft)
    return spectrogram

def getMFCC(data, rate, windowFFT, windowMFCC):
    return librosa.feature.mfcc(y=np.array(data), sr=rate, n_mfcc= 12, hop_length=windowMFCC, n_fft = windowFFT)

def compare2Params(row1,row2):
    return distance.euclidean(row1,row2)

def matrixCompare2files(data1, data2):
    result = np.zeros((data1.shape[1], data2.shape[1]))
    for i in range(data1.shape[1]):
        for j in range(data2.shape[1]):
            result[i,j] = compare2Params(data1[:,i], data2[:,j])
    return result

def DTW(matrix):
    result = np.zeros(matrix.shape)
    for i in range(0,result.shape[0]):
        if(i>0):
            result[i,0]= math.inf
        
        for j in range(result.shape[1]):
            if(j>0):
                if(i==0):
                    result[i,j]= math.inf
                else:
                    result[i,j]= matrix[i,j]+min(min(result[i,j-1],result[i-1,j]),result[i-1,j-1])+1
        
    return result

def decrement(i,j,elem,idx):
    elem = np.delete(elem, idx)
    
    if(idx == 0):
        if(i > 0):
            i = i - 1
        else:
            idx = np.argmin(elem)            
            return decrement(i,j,elem,idx)
    if(idx == 2):
        if(j > 0):
            j = j - 1
        else:
            idx = np.argmin(elem)
            return decrement(i,j,elem,idx)
    if(idx == 1):
        if(j > 0 and i > 0):
            i = i-1
            j = j-1
        else:
            idx = np.argmin(elem)
            return decrement(i,j,elem,idx)
    return i,j,idx
	
def shortPath(matrix):
    result = np.zeros(matrix.shape)
    i = matrix.shape[0] - 1
    j = matrix.shape[1] - 1
    xMax = i
    yMax = j
    value = 0
    last = 1
    step = 0
    while(i > 0 and j > 0):
        result[i,j] = 1
        value = value + math.pow(((yMax/xMax*i+j)/math.sqrt((yMax/xMax)*(yMax/xMax)+1)),2)
        a = matrix[max(i-1,0),j]
        b = matrix[max(i-1,0),max(j-1,0)]
        c = matrix[i,max(j-1,0)]
        idx = np.argmin(np.array([a,b,c]))
        i,j,idx = decrement(i,j,np.array([a,b,c]),idx)
        if(idx != 1):
            if(idx == last):
                step = step + 1
            else:
                step = 0
        else:
            step = 0
        last = idx
        
        value= value + math.pow(step,4)
        
    while(i > 0 ):  
        result[i,j] = 1
        i = i - 1
        step = step + 1
        
    while(j > 0 ):  
        result[i,j] = 1
        j = j - 1
        step = step + 1
    result[0,0]=1
	
    return result, value

def my_custom_norm(x, y):
    return (x * x) + (y * y)

def CompareFilesBib(mfcc1, mfcc2):
    dist, cost, acc, path = dtw(mfcc1.T, mfcc2.T, dist=lambda x, y: norm(x - y, ord=1))
    return cost,path,dist


def CompareFiles(mfcc1, mfcc2):
    sim = matrixCompare2files(mfcc1,mfcc2)
    dtw = DTW(sim)
    path,value = shortPath(dtw)
    return dtw,path,dtw[dtw.shape[0]-1,dtw.shape[1]-1]/(dtw.shape[0]+dtw.shape[1])#value/np.sum(path)*

def loadFile(fileName):
    wav = wave.open (fileName, "r")
    (nchannels, sampwidth, framerate, nframes, comptype, compname) = wav.getparams ()    
    frames = wav.readframes (nframes * nchannels)
    out =  struct.unpack_from ("%dh" % nframes * nchannels, frames)
    return out,framerate

def createPattern():
    files = glob.glob("wav\*.wav")
    for file in files:
        out,framerate = loadFile(file)
        mfcc = getMFCC(out,framerate,windowFFT,windowMFCC)
        file = file.replace("cyfry\\","").replace(".WAV","")
        np.save("pattern\\"+file+'.npy', mfcc)
        
def loadPattern():
    files = glob.glob("pattern\*.npy")
    patterns= []
    for file in files:
        data = np.load(file)
        file = file.replace("pattern\\","").replace(".npy","")
        patterns.append([file,data,int(file[len(file)-2])])
    return patterns

def compareWithPattern(filename,pattern):
    out,rate = loadFile(filename)
    data = getMFCC(out,rate,windowFFT,windowMFCC)
    values = []
    for p in pattern:
        dtw,path,value = CompareFiles(data,p[1])        
        values.append([dtw,path,value])
    return values

def comparePatternToALL(paternToTest,AllPatterns):
    values = []
    for p in AllPatterns:
        dtw,path,value = CompareFiles(paternToTest,p[1])

        values.append([dtw,path,value])
    return values


def main():
    wav = wave.open ('wav\Band_of_Blacky_Ranchette-Mope-a-Long_Rides_Again.wav', "r")
    (nchannels, sampwidth, framerate, nframes, comptype, compname) = wav.getparams ()
    
    frames = wav.readframes (nframes * nchannels)
    out = struct.unpack_from ("%dh" % nframes * nchannels, frames)
 
    NFFT = 1024

    plt.figure(1)
    plt.title('Signal')
    plt.plot(out)
    #plt.show()
    
    Time = np.linspace(0, len(out)/nframes, num=len(out))

    plt.figure(2)
    plt.title('Time Wave...')
    plt.plot(Time,out)
    #plt.show()
    
    sp = np.fft.fft(out)
    
    plt.figure(3)
    #plt.pcolormesh(times, frequencies, spectogram)
    #plt.ylabel('Frequency [Hz]')
    #plt.xlabel('Time [sec]')
    plt.plot(sp)
    plt.figure(4)
    spectrum , freqs, bins, im = plt.specgram(out, NFFT=NFFT, Fs=framerate, noverlap=1000, window = window(1,NFFT,1))
    num_ceps  = 12
    mfcc = librosa.feature.mfcc(y=np.array(out), sr=framerate, n_mfcc= 12,hop_length=int(framerate/100),n_fft =512)
    mfcc2 = librosa.feature.mfcc(y=np.array(out), sr=framerate, n_mfcc= 12,hop_length=int(framerate/128),n_fft =512)
    plt.figure(5)
    librosa.display.specshow(mfcc, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    plt.show()
    dtw,path,value = CompareFiles(mfcc,mfcc2)
    
    dtw[path==1] = dtw.max()+10
    plt.figure(6)
    plt.imshow(dtw)
    
    
def KNN(valuesUtility,k,values,patterns2):
    indexs = np.argpartition(valuesUtility,k)
    maxVal = np.max(valuesUtility)
    groupValues =np.zeros(10) + maxVal
    group  =[]
    for idx in indexs[0:k]:
        groupValues[patterns2[idx][2]] = min(groupValues[patterns2[idx][2]],values[idx][2])
        group.append(patterns2[idx][2])
        
    
    groupCount = Counter(group).most_common(2)
    if(len(groupCount)>1):
        if(groupCount[0][1]==groupCount[1][1]):
            if(groupValues[groupCount[0][0]] < groupValues[groupCount[0][0]]): 
                return groupCount[0][0]
            else:
                groupCount[1][0]
            
    
    return groupCount[0][0]
def TestAllFilesSlow():
    correct = 0
    AllPatterns = loadPattern()
    
    files = glob.glob("wav\*.wav")
    for file in files:
        patterns2 = []
        fileName = file.replace("wav\\","").replace(".WAV","")
        for pattern in AllPatterns:
            if pattern[0]  != fileName:
                patterns2.append(pattern)
            
        values = compareWithPattern('wav\\'+fileName+'.wav',patterns2)
        valuesUtility = np.array([item[2] for item in values])
        indexsMin = np.argmin(valuesUtility)
        decision = patterns2[indexsMin][2]
        orginalDecision = fileName[len(fileName)-2]
        if(decision == orginalDecision):
            correct = correct+1
        print(orginalDecision,decision)
    print(correct/len(files)*100)

def TestAllFilesFast():
    correctKNN1 = 0.0
    correctKNN3 = 0.0
    correctKNN5 = 0.0
    correctOUT = 0.0
    AllPatterns = loadPattern()
    
    files = glob.glob("wav\*.wav")
    for file in files:
        patterns2 = []
        fileName = file.replace("wav\\","").replace(".WAV","")
        
        for pattern in AllPatterns:
            if pattern[0]  != fileName:
                patterns2.append(pattern)
            else:
                patternToTest = pattern
            
        values = comparePatternToALL(patternToTest[1],patterns2)
        valuesUtility = np.array([item[2] for item in values])
        indexsMin = np.argmin(valuesUtility)
        decision = patterns2[indexsMin][2]
        orginalDecision = int(fileName[len(fileName)-2])
        knnDecision5 = KNN(valuesUtility,5,values,patterns2)
        knnDecision3 = KNN(valuesUtility,3,values,patterns2)
        
        decisionOutput = Counter([decision,knnDecision3,knnDecision5]).most_common(1)[0][0]
        if(decision == orginalDecision):
            correctKNN1 = correctKNN1+1.0            
        if(knnDecision3 == orginalDecision):
            correctKNN3 = correctKNN3+1.0
        if(knnDecision5 == orginalDecision):
            correctKNN5 = correctKNN5+1.0
        if(decisionOutput == orginalDecision):
            correctOUT = correctOUT+1.0
        print(orginalDecision,decision,knnDecision3,knnDecision5,decisionOutput)
    print(1,correctKNN1/len(files)*100.0)
    print(3,correctKNN3/len(files)*100.0)
    print(5,correctKNN5/len(files)*100.0)
    print("OUT",correctOUT/len(files)*100.0)
    
def testOne():
    AllPatterns = loadPattern()
    patterns2 = []
    fileName = "Band_of_Blacky_Ranchette-Mope-a-Long_Rides_Again"
    for pattern in AllPatterns:
        if pattern[0]  != fileName:
            patterns2.append(pattern)
        else:
            patternToTest = pattern
            
    values = comparePatternToALL(patternToTest[1],patterns2)
    valuesUtility = np.array([item[2] for item in values])
    indexsMin = np.argmin(valuesUtility)
    decision = patterns2[indexsMin][2]
    orginalDecision = int(fileName[len(fileName)-2])
    knnDecision5 = KNN(valuesUtility,5,values,patterns2)
    knnDecision3 = KNN(valuesUtility,3,values,patterns2)
    
    decisionOutput = Counter([decision,knnDecision3,knnDecision5]).most_common(1)[0][0]

    print("orginal decision",orginalDecision)
    print("Decision",decisionOutput )

    dtw = values[indexsMin][0]
    path = values[indexsMin][1]
    dtw[path==1] = dtw.max()+10
    plt.figure(6)
    plt.imshow(dtw)
    
if __name__ == "__main__":
    #createPattern()
    #TestAllFilesFast()
    testOne()
    
