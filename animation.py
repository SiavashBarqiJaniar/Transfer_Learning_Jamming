"""
Copyright 2023 Siavash Barqi Janiar

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

scenario = input("Scenario? ")#"sweep2"
classifier = input("Classifier? ") #freeze

# figure
plt.figure(1, dpi = 50)
plt.style.use('fivethirtyeight')
plt.tight_layout()
ax1 = plt.subplot(2,1,1)

ax2 = plt.subplot(2,1,2)
ax2.set_ylim(bottom=0, top=10)

def avg(arr):
    window = 40
    avgArr = []
    for j in range(len(arr)):
        if j < window-1:
            avgArr.append( sum(arr[:j+1]) / (j + 1) )
        else:
            avgArr.append( sum(arr[j - (window-1):j+1]) / window )
    return avgArr

def readcsv(add):
    try:
        df = pd.read_csv(add)
        arr = df.to_numpy()
        return arr
    except:
        return []

def animate(i):
    try:
        arr1  = readcsv(f'{scenario}_testThroughputsRL.csv')
    except:
        arr1 = []
    try:
        arrStates = readcsv(f'{scenario}_states.csv')
    except:
        arrStates = []
    try:
        arr2  = readcsv(f'{scenario}ThroughputsTLfreeze.csv')
    except:
        arr2 = []
    try:
        arr3  = readcsv(f'{scenario}ThroughputsTLclassifier.csv')
    except:
        arr3 = []
    try:
        arr4 = readcsv(f'{scenario}ThroughputsTLwinit.csv')
    except:
        arr4 = []
    # to show different epsilon plots
    # arr1 = readcsv(f'{scenario}EP9ThroughputsTL{classifier}.csv')
    # arr2 = readcsv(f'{scenario}EP7ThroughputsTL{classifier}.csv')
    # arr3 = readcsv(f'{scenario}EP5ThroughputsTL{classifier}.csv')
    # arr4 = readcsv(f'{scenario}EP3ThroughputsTL{classifier}.csv')
    # arr5 = readcsv(f'{scenario}EP1ThroughputsTL{classifier}.csv')

    # average
    avgArr1 = avg(arr1)
    avgArr2 = avg(arr2)
    avgArr3 = avg(arr3)
    avgArr4 = avg(arr4)
    try:
        avgArr5 = avg(arr5)
    except:
        avgArr5 = []
    #plt.subplot(2, 1, 1)
    ax1.cla()
    ax1.set_title(f'{scenario} , {classifier}')
    #ax1.plot(range(len(arr1)), arr1, color='g', label='RL')
    ax1.plot(range(len(avgArr1)), avgArr1, color='r', label='RL avg')
    #ax1.plot(range(len(arr2)), arr2, color='b', label='TL-F')
    ax1.plot(range(len(avgArr2)), avgArr2, color='grey', label='TL-F avg')
    #ax1.plot(range(len(arr3)), arr3, color='brown', label='TL-C')
    ax1.plot(range(len(avgArr3)), avgArr3, color='orange', label='TL-C avg')
    #ax1.plot(range(len(arr4)), arr4, color='y', label='TL-WI')
    ax1.plot(range(len(avgArr4)), avgArr4, color='m', label='TL-WI avg')
    ax1.plot(range(len(avgArr5)), avgArr5, color='pink', label='e = 0.1')
    ax1.legend()
    #plt.subplot(2,1,2)
    
    ax2.cla()
    repData = arrStates.copy()
    repData[repData == 1] = 0
    repData[repData == -2] = 0
    for i in range(len(repData)):
        for j in range(len(repData[0])):
            if repData[i,j] == -1:
                repData[i,j] = j+1
    ax2.plot(repData, '.', color='red')#, label='Jam')
    repData = arrStates.copy()
    repData[repData == 1] = 0
    repData[repData == -1] = 0
    for i in range(len(repData)):
        for j in range(len(repData[0])):
            if repData[i,j] == -2:
                repData[i,j] = j+1
    ax2.plot(repData, '.', color='black')#, label='Disruption')
    repData = arrStates.copy()
    repData[repData == -1] = 0
    repData[repData == -2] = 0
    for i in range(len(repData)):
        for j in range(len(repData[0])):
            if repData[i,j] == 1:
                repData[i,j] = j+1
    ax2.plot(repData, '.', color='green')#, label='Signal')
    
    """
    plt.plot(range(len(arr2)), arr2, label='throughputs 2')
    plt.plot(range(len(arr3)), arr3, label='throughputs TL')
    plt.plot(range(len(arr4)), arr4, label='throughputs TL freezing')
    plt.plot(range(len(arr5)), arr5, label='throughputs TL from scratch')
    """
    #plt.legend()

ani = FuncAnimation(plt.gcf(), animate, interval=20000) #10
plt.show()