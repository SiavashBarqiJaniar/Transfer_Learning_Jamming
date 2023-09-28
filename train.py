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

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input
from keras.models import load_model
from random import randint
import pandas as pd
from tensorflow.keras.optimizers import RMSprop
import time

# hyper parameters
scenario = input("scenario? ")
flagRL = True
flagTLfreeze = False
flagTLwinit = False
flagTLclassifier = False
noSimulationTimeSlots = 100000 # 100,000 == 10,000 packets
ct = 100
t2 = 300 #1000
noChannels = 9 #9
noActions = noChannels
signalDuration = 10
if scenario[0] == 'd':
    observationMemory = 20
else:
    observationMemory = 20
stateInSingleTimeSlot = np.zeros(noChannels, dtype=int)
stateAfterSendingSignal = np.ndarray(shape=(signalDuration, noChannels), dtype=int)
action = np.zeros(noActions, dtype=int)
jammerAction = np.zeros(noActions, dtype=int)
trajectory = np.array( np.shape(stateAfterSendingSignal)[0] + noActions + np.shape(stateAfterSendingSignal)[0] + 1 , dtype=int ) # (s, a, s', r)
inputToDNN = [] # states
dnnTargets = [] # [Qs , (r + gamma*sumRewards)
collisionFlag = False
switched = False
noCollisions = 0
bufferStateInSingleTimeSlot = []
bufferStateAfterSendingSignal = [] # [ [stateAfterSendingSignal], [stateAfterSendingSignal], [stateAfterSendingSignal], ... ]
bufferActionInSingleTimeSlot = []
bufferActionAfterSendingSignal = []
rewardDiscountedSum = 0.0
Bsize = 10
B = [[] for x in range(Bsize)]
rBuffer = [[] for x in range(Bsize)]
aBuffer = [[] for x in range(Bsize)]
QBuffer = []
if scenario == 'd':
    gamma = 0.1 #0.6
else:
    gamma = 0.1 #0.6
r = 0
currentThroughput = []
# e-policy
epsMin = 0 #.05
if flagRL:
    epsilon = .9
else:
    epsilon = .1 #.9*normal KL #.9
epsilonDecay = .997 #.9994 #.9985 #.9856 #.997
epsilonSub = 0.00010625

def createSourceModel():
    #  100 time , 5 channels
    inputs = Input(shape=(observationMemory*signalDuration, noChannels))
    x = layers.LSTM(128)(inputs)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(16, activation="relu")(x)
    #x = layers.Dense(16, activation="relu")(x)
    outputs = layers.Dense(noActions, activation="softmax")(x)
    # make
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy"]) # optimizer=RMSprop(lr=0.02)
    #model.fit(x_train, y_train, batch_size=128, epochs=2, validation_split=0.1)
    return model

# create the source model
try:
    sourceModel = load_model('RLsweep.h5')
    for l in sourceModel.layers:
        l.trainable = False
    # to see the name of source model layers
    #sourceModel.summary()
    if flagTLwinit:
        targetModel = load_model('RLsweep.h5')
        targetModelPrime = load_model('RLsweep.h5')
    elif flagTLclassifier:
        targetModel = sourceModel
        targetModelPrime = sourceModel
    # Using and training a DNN from the scratch
    elif flagRL:
        targetModel = createSourceModel()
        targetModelPrime = createSourceModel()
    else:
        # pass the second to last layer to a new (trainable) layer
        #x = layers.Flatten()(sourceModel.layers[-2].output)
        outputs = layers.Dense(noActions, activation='softmax', name='dense_output')(sourceModel.layers[-2].output)

        # make the model
        targetModel = keras.Model(inputs=sourceModel.inputs, outputs=outputs)
        targetModel.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
        
        targetModelPrime = keras.Model(inputs=sourceModel.inputs, outputs=outputs)
        targetModelPrime.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
        
        targetModel.set_weights(sourceModel.get_weights())
        targetModelPrime.set_weights(sourceModel.get_weights())

    # target model summary
    targetModel.summary()
    # confirm that the model is created successfully
    print('Loaded the model successfully.')

except:
    print("!!! Error creating source model. !!!")

def updateSourceModel():
    # training the target model
    # critic
    rewardDiscountedSum += r + gammaCritic*rewardDiscountedSum
    # actor: chosen RL alg = online Q-learning (iteration)
    nextState = bufferStateAfterSendingSignal[-observationMemory:]
    concatenatedNextState = np.concatenate( np.concatenate( [nextState] ) )
    concatenatedNextState = concatenatedNextState[np.newaxis]
    [Q, c] = targetModel.predict( concatenatedNextState + np.ones_like(concatenatedNextState) )
    ###Q' = predict(stateAfterSendingSignal)
    Q = Q[0]
    y = Q.copy()
    y[intAction-1] = r + gammaActor * max(Q) # gammaActor*Q'[argmax(Q)]
    inputToDNN = bufferStateAfterSendingSignal[-observationMemory-1:-1]
    inputToDNN = np.concatenate( np.concatenate( [inputToDNN] ) )
    inputToDNN = inputToDNN[np.newaxis]
    [targetQ, cc] = targetModel.predict(inputToDNN)
    targetQ = targetQ[0]
    targetQ[intAction-1] = y[intAction-1].copy()
    # training
    ccc = np.ndarray(shape=(1,1), dtype=float)
    ccc[0, 0] = rewardDiscountedSum
    temp2 = np.ndarray(shape=( 1,np.shape(targetQ)[0] ), dtype=float)
    temp2[0] = targetQ
    targetModel.fit(inputToDNN, [temp2, ccc], batch_size=1, epochs=1) # , validation_split=0.1)

def takeAction(s = np.zeros(noChannels, dtype=int), alg='aloha'):
    if alg == 'aloha':
        return randint(1, noActions)
    elif alg == 'rl':
        global epsilon
        epsilon = max(epsMin, epsilon)
        if np.random.ranf() <= epsilon: # returns a float [0.0, 1.0)
            epsilon *= epsilonDecay
            #epsilon -= epsilonSub
            return randint(1, noActions)
        else:
            # to check source model still works, I first test sourceModel
            predictedActionDistribution = targetModel.predict(s)
            QBuffer.append(predictedActionDistribution[0].copy())
            #predictedAction = np.argmax(predictedActionDistribution[0]) + 1
            predictedAction = np.random.choice(noActions, p=predictedActionDistribution[0]) + 1
            epsilon *= epsilonDecay
            #epsilon -= epsilonSub
            return predictedAction

def normalize(vec):
    vec[vec < 0] = 0
    for i in range(len(vec)):
        vec[i] /= sum(vec)
    return vec

def jamSweep(t):
    jAction = []
    if t%24 <= 6:
        jAction.append(1)
    if t%24 <= 8 and t%24 >= 2:
        jAction.append(2)
    if t%24 <= 10 and t%24 >= 4:
        jAction.append(3)
    if t%24 <= 12 and t%24 >= 6:
        jAction.append(4)
    if t%24 <= 14 and t%24 >= 8:
        jAction.append(5)
    if t%24 <= 16 and t%24 >= 10:
        jAction.append(6)
    if t%24 <= 18 and t%24 >= 12:
        jAction.append(7)
    if t%24 <= 20 and t%24 >= 14:
        jAction.append(8)
    if t%24 <= 22 and t%24 >= 16:
        jAction.append(9)
    return jAction

def jamSweep2(t):
    jAction = []
    if t%30 <= 8:
        jAction.append(1)
    if t%30 <= 10 and t%30 >= 2:
        jAction.append(2)
    if t%30 <= 12 and t%30 >= 4:
        jAction.append(3)
    if t%30 <= 14 and t%30 >= 6:
        jAction.append(4)
    if t%30 <= 16 and t%30 >= 8:
        jAction.append(5)
    if t%30 <= 18 and t%30 >= 10:
        jAction.append(6)
    if t%30 <= 20 and t%30 >= 12:
        jAction.append(7)
    if t%30 <= 22 and t%30 >= 14:
        jAction.append(8)
    if t%30 <= 24 and t%30 >= 16:
        jAction.append(9)
    return jAction

def jamD(t):
    jAction = []
    if (t%100) <= 50 and t%100 != 0:
        jAction = [1, 2, 3, 4, 5, 6]
    else:
        jAction = [1, 2, 6, 7, 8, 9]
    return jAction

def jamIntel(t, actions):
    jAction = []
    maxx = [0 for x in range(noActions)]
    for a in actions:
        maxx[a - 1] += 1
    for x in range(6):
        if max(maxx) != 0:
            imax = maxx.index(max(maxx))
            jAction.append(imax + 1)
            maxx[imax] = 0
        else:
            select = np.random.randint(noChannels) + 1
            if select in jAction:
                x += 1
            else:
                jAction.append(select)
    return jAction

def jamDist():
    jAction = []
    #jammingDistribution = [.01, .06, .06, .2, .2, .2, .2, .06, .01] #[.1, .1, .25, 0.2, 0.05, 0.1, 0.1, 0.05, 0.05]
    #jammingDistribution = [.15, .1, .25, .1, .05, .05, .1, .15, .05]
    # jammingDistribution = [.05, .05, .05, .05, .1, .05, .4, .1, .15]
    jammingDistribution = [.016, .23, .016, .23, .016, .23, .016, .23, .016]
    chosenChannels = np.random.choice(noActions, p=jammingDistribution, size=6, replace = False)
    for i in range(len(chosenChannels)):
        chosenChannels[i] += 1
    return chosenChannels.tolist()

# creating data

# start the communication network simulation
print('\n\n\nWireless network simulation started.')
for timeSlot in range(1, noSimulationTimeSlots + 1):
    # the beginning of the time slot
    stateInSingleTimeSlot = np.zeros(noChannels, dtype=int)
    action = np.zeros(noActions, dtype=int)
    jammerAction = np.zeros(noActions, dtype=int)
    if timeSlot % 10 == 1:
        if timeSlot <= signalDuration*observationMemory:# or (not flagRL and timeSlot <= 1000):
            intAction = takeAction('aloha')
        else:
            temp = np.concatenate( np.concatenate( [bufferStateAfterSendingSignal[-observationMemory:]] ) )
            temp = temp[np.newaxis]
            previousIntAction = intAction
            intAction = takeAction(temp, 'rl')
            if timeSlot % 1000 == 1:
                print('*** action taken:', intAction)
            else:
                pass
            if intAction == previousIntAction:
                switched = True
            else:
                switched = False
    else:
        intAction = bufferActionInSingleTimeSlot[-1]
    action[intAction-1] = 1
    # Jammer Action
    if scenario[:5] == "intel":
        #if timeSlot <= 100:
        #    intJammerAction = takeAction('aloha')
        if timeSlot % 100 == 1:
            intJammerAction = jamIntel(timeSlot, bufferActionAfterSendingSignal[-10:]) #jamSweep(timeSlot)
    elif scenario == 'sweep':
        intJammerAction = []
        intJammerAction = jamSweep(timeSlot)
    elif scenario[:6] == 'sweep2':
        intJammerAction = []
        intJammerAction = jamSweep2(timeSlot)
    elif scenario[0] == 'd':
        intJammerAction = jamD(timeSlot)
    elif scenario[:4] == 'prob':
        if timeSlot % 100 == 1:
            intJammerAction = jamDist()
    #intJammerAction = randint(1, noActions)
    for intjaction in intJammerAction:
        jammerAction[intjaction-1] = -1 #2
    
    #the end of the time slot
    
    if intJammerAction.count(intAction) > 0:
        collisionFlag = True
    else:
        pass
    stateInSingleTimeSlot = action + jammerAction
    if collisionFlag:
        stateInSingleTimeSlot[intAction-1] = -2#-1
    else:
        pass
    stateAfterSendingSignal[(timeSlot % 10) - 1] = stateInSingleTimeSlot.copy()
    bufferStateInSingleTimeSlot.append(stateInSingleTimeSlot)
    bufferActionInSingleTimeSlot.append(intAction)
    if timeSlot % 10 == 0:
        if collisionFlag:
            # signal disrupted
            noCollisions += 1
            r = -.1 # 0 #-1 #-.5 #-1
            stateAfterSendingSignal[stateAfterSendingSignal == 1] = -2
        #elif switched:
        #    r = 1#.8
        else:
            r = 1
        #if switched:
        #    r -= .2
        #else:
        #    pass
        collisionFlag = False
        bufferStateAfterSendingSignal.append(stateAfterSendingSignal.copy())
        bufferActionAfterSendingSignal.append(intAction)
        if timeSlot >= signalDuration*observationMemory:
            B[int(timeSlot/signalDuration) % Bsize] = bufferStateAfterSendingSignal[-observationMemory:]
            rBuffer[int(timeSlot/signalDuration) % Bsize] = r
            aBuffer[int(timeSlot/signalDuration) % Bsize] = intAction
        else:
            pass
        
        if timeSlot >= signalDuration*observationMemory and timeSlot >= Bsize*observationMemory*signalDuration:
            samples = np.random.choice(Bsize, size=int(Bsize/2))
            #for sample in samples:
            #nextState = B[sample]
            #r = rBuffer[sample]
            #intAction = aBuffer[sample]
            # training the target model
            # chosen RL alg = online Q-learning (iteration)
            nextState = bufferStateAfterSendingSignal[-observationMemory:]
            concatenatedNextState = np.concatenate( np.concatenate( [nextState] ) )
            concatenatedNextState = concatenatedNextState[np.newaxis]
            Q = targetModel.predict( concatenatedNextState + np.ones_like(concatenatedNextState) )
            Qp = targetModelPrime.predict( concatenatedNextState + np.ones_like(concatenatedNextState) )
            #print('***', Q, Qp)
            ###Q' = predict(stateAfterSendingSignal)
            Q = Q[0]
            Qp = Qp[0]
            y = Q.copy()
            y[intAction-1] = r + gamma * Qp[np.argmax(Q)] # max(Q) Bellman Eq.
            inputToDNN = bufferStateAfterSendingSignal[-observationMemory-1:-1]
            inputToDNN = np.concatenate( np.concatenate( [inputToDNN] ) )
            inputToDNN = inputToDNN[np.newaxis]
            targetQ = targetModel.predict(inputToDNN)
            targetQ = targetQ[0]
            targetQ[intAction-1] = y[intAction-1].copy()
            #targetQ = normalize(targetQ)
            # training
            temp2 = np.ndarray(shape=( 1,np.shape(targetQ)[0] ), dtype=float)
            temp2[0] = targetQ
            if not flagTLclassifier:
                targetModel.fit(inputToDNN, temp2, batch_size=1, epochs=1, verbose=0) # , validation_split=0.1)
            if timeSlot % 100 == 1:
                targetModelPrime.set_weights(targetModel.get_weights())
            #if timeSlot >= 10000 and timeSlot % 1000 == 0:
            #    gamma += 0.1
            #    gamma = min(gamma, 0.9)

    # verbose
    if timeSlot % t2 == 0:
        t1 = timeSlot - t2
        a1 = t1*(1 - ct/100)/10
        print(f'time slot {timeSlot} has been passed. epsilon: {epsilon}')
        ct = (1 - ( (noCollisions*10) / timeSlot ) ) * 100
        print(f'Totoal throughput: {ct}')
        a2 = noCollisions - a1
        c2 = (1 - a2*10/t2)*100
        print('Current throughput:', c2)
        print('Discount factor:', gamma)
        currentThroughput.append(c2)
        dfCurrentThruput = pd.DataFrame(currentThroughput)
        if flagTLfreeze:
            dfCurrentThruput.to_csv(f'{scenario}ThroughputsTLfreeze.csv', index=False)
        elif flagTLclassifier:
            dfCurrentThruput.to_csv(f'{scenario}ThroughputsTLclassifier.csv', index=False)
        elif flagTLwinit:
            dfCurrentThruput.to_csv(f'{scenario}ThroughputsTLwinit.csv', index=False)
        else:
            dfCurrentThruput.to_csv(f'{scenario}ThroughputsRL.csv', index=False)
        dfStates = pd.DataFrame( np.concatenate(bufferStateAfterSendingSignal) )
        dfStates.to_csv(f'{scenario}_states.csv', index=False)
        dfQ = pd.DataFrame(QBuffer)
        dfQ.to_csv('Qvalues.csv')
        dfActions = pd.DataFrame( bufferActionAfterSendingSignal )
        dfActions.to_csv(f'{scenario}_actions.csv', index=False)
    if timeSlot % 1000 == 0 and flagRL:
        targetModel.save(f"RL{scenario}.h5")

# printing information
print('throughput:', (1 - ( (noCollisions*10) / noSimulationTimeSlots ) ) * 100)
print('no collisions:', noCollisions)
print('The end of the wireless network simulation.\n\n\n')

