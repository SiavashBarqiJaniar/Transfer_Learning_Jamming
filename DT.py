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

import csv
import os
import numpy as np
import math
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# Visualization
from sklearn import tree
import IPython, graphviz, re, math
from matplotlib import pyplot as plt
from random import shuffle
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", default='trajectories/game99.csv')
parser.add_argument("--val_size", default=0.1)
parser.add_argument("--test_size", default=0.1)
parser.add_argument("--bs", default=512)
parser.add_argument("--lr", default=0.0001)
parser.add_argument("--num_epochs", default=100)
parser.add_argument("--game", default=99)
args = parser.parse_args()

###################################
# Hyperparameters
###################################
A = 9
scenario = input('Scenario? ')

def readcsv(add):
    try:
        df = pd.read_csv(add)
        arr = df.to_numpy()
        return arr
    except:
        return []

def get_instances():
    verbous = True
    dataset = []
    x = []
    y = []
    states = readcsv(f'{scenario}_testStates.csv')
    actions = readcsv(f'{scenario}_testActions.csv')
    print('***', np.shape(states), type(states))
    print('***', np.shape(actions), type(actions))
    no_instances = len(actions)
    channels = []
    for i in range(int(no_instances)):
        channel = [[] for i in range(A)]
        for j in range(A):
            channel[j] = states[i * 10:(i + 1) * 10, j]
        chs = []
        for ch in channel:
            if np.count_nonzero(ch == -2) > 0:
                chs.append( [1, 0, 0, 0, 1, 1, 1] ) # disrupted
                # .., 1, 1, 1] means .., disrupted/jammed, dis/user, dis/idle]
            elif np.count_nonzero(ch == -1) > 0:
                chs.append( [0, 1, 0, 0, 1, 0, 0] ) #'occupied by the jammer')
                #print('*** -1', ch.count(-1), ch)
            elif np.count_nonzero(ch == 1) == len(ch):
                chs.append( [0, 0, 1, 0, 0, 1, 0] ) #'occupied by the user')
                #print('*** 1', ch.count(1), ch)
            elif np.count_nonzero(ch == 0) == len(ch):
                chs.append( [0, 0, 0, 1, 0, 0, 1] ) #'idle')
                #print('*** 0', ch)
        assert len(chs) == len(channel)
        channels.append( np.concatenate(chs) )
        #print('*** result:', chs)

    # Creating dataset
    for i in range(no_instances):
        x.append(channels[i])
        y.append(actions[i,0])
        #dataset.append((x, y))
    print("Complete dataset", len(x), len(y), x[0])
    return x, y
    
def train(x, y):
    model = DecisionTreeClassifier(criterion="gini")

    imp = {}
    for i in range(200):
        # Shuffling instances
        z = list(zip(x, y))
        shuffle(z)
        x_, y_ = zip(*z)

        # fitting the model
        model.fit(x_, y_)

        # get importance
        importance = model.feature_importances_
        if i == 0:
            imp = importance.copy()
        else:
            imp += importance.copy()
    temp = imp.copy()/200

    return model, temp

def draw_tree(t, col_names, size=9, ratio=0.5, precision=3):
    """ Draws a representation of a random forest in IPython.
    Parameters:
    -----------
    t: The tree you wish to draw
    df: The data used to train the tree. This is used to get the names of the features.
    """
    s = tree.export_graphviz(t, out_file=None, feature_names=col_names, filled=True,
                      special_characters=True, rotate=True, precision=precision)
    IPython.display.display(graphviz.Source(re.sub('Tree {',
       f'Tree {{ size={size}; ratio={ratio}',s)))

def find_probability_distribution(i, x, y):
    counter = [0 for m in range(A)]
    for j in range(len(y)):
        if (x[j] == x[i]).all():
            counter[y[j] - 1] += 1
    tot_counter = sum(counter)
    for m in range(A):
        counter[m] /= tot_counter
    return counter

def MSE(prob, pred):
    summ = 0.0
    for j in range(len(pred)):
        summ += pow( (pred[j] - prob[j]), 2 )
    summ /= A
    return summ

def MAE(prob, pred):
    summ = 0.0
    for j in range(len(pred)):
        summ += abs(pred[j] - prob[j])
    summ /= A
    return summ



if __name__ == '__main__':
    """
    HIPER-PARAMETERS
    """
    iterations = 5
    test = False


    #mdl = LogisticRegression(solver='liblinear')
    total_scores = [0.0 for j in range(A)]
    if False:#not test:
        for j in range(iterations):
            print('----------------------------------')
            print(f'-------------- {j} -----------------')
            if not test:
                x, y = get_instances(j, j)
            else:
                x, y = get_instances(50+j, 50+j)
            print('data:', len(x), len(y))

            mdl, imps = train(x, y)

            # summarize feature importance
            for i,v in enumerate(imps):
                print('Feature: %0d, Score: %.5f' % (i,v))
                total_scores[i] += v
            # plot feature importance
            plt.bar([m for m in range(len(imps))], imps)
            plt.show()

    if False:
        if not test:
            x, y = get_instances(0, iterations-1)
        else:
            x, y = get_instances(50, 50)
        print('data:', len(x), len(y))
        mdl, imps = train(x, y)
        for i,v in enumerate(total_scores):
            v /= iterations
            total_scores[i] = v
            print('Feature: %0d, Score: %.5f' % (i,v))
        plt.bar([m for m in range(len(total_scores))], total_scores)
        plt.show()
    
    ### Plotting the Tree
    if False:
        model = DecisionTreeClassifier(criterion="gini")
        for j in range(iterations):
            x, y = get_instances(iterations-j-1, iterations-j-1)
            for i in range(200):
                # Shuffling instances
                z = list(zip(x, y))
                shuffle(z)
                x, y = zip(*z)

                #fitting the tree model
                model.fit(x, y)
            print('data:', len(x), len(y))

    MAEMemory = []
    RMSEMemory = []
    for c in range(10):
        x, y = get_instances()
        xTrain = x[:int(len(x) * .8)].copy()
        yTrain = y[:int(len(y) * .8)].copy()
        print('data:', len(xTrain), len(yTrain))
        model, imps = train(xTrain, yTrain)
        imps = model.feature_importances_
        verbose = False
        if verbose:
            for i,v in enumerate(imps):
                print('Feature: %0d, Score: %.5f' % (i,v))

        ###### Accuracy
        xTest = x[int(len(x) * .8):].copy()
        yTest = y[int(len(y) * .8):].copy()
        print('data:', len(xTest), len(yTest))
        #accuracy = model.score(xTest, yTest)
        #accuracy = accuracy_score(yTest, y_pred)
        #print('Accuracy: %.2f' % (accuracy*100))
        
        y_pred = model.predict_proba(xTest)
        errors_MSE = []
        errors_MAE = []
        for k in range(len(yTest)):
            prob_dist = find_probability_distribution(k, xTest, yTest) # (A)
            errors_MSE.append(MSE(prob_dist, y_pred[k]))
            errors_MAE.append(MAE(prob_dist, y_pred[k]))
        error_RMSE = math.sqrt(np.average(errors_MSE))
        error_MAE = np.average(errors_MAE)
        MAEMemory.append(error_MAE)
        RMSEMemory.append(error_RMSE)
        print('error MSE', error_RMSE)
        print('error MAE', error_MAE)
        print('Accuracy MSE: %.2f' % ((1 - error_RMSE)*100))
        print('Accuracy MAE: %.2f' % ((1 - error_MAE)*100))
        with open(f'report/{scenario}.txt', 'a') as f:
            f.write('Accuracy MSE: %.2f' % ((1 - error_RMSE)*100) + '\r\n')
            f.write('Accuracy MAE: %.2f' % ((1 - error_MAE)*100) + '\r\n')
    with open(f'report/{scenario}.txt', 'a') as f:
        f.write('Accuracy MAE avg: %.2f' % ((1 - np.average(MAEMemory))*100) + '\r\n')
        f.write('Accuracy RMSE avg: %.2f' % ((1 - np.average(RMSEMemory))*100) + '\r\n')

    no_conditions = int(len(x[0])/A)
    features = []
    for i in range(A*no_conditions):
        if i%no_conditions == 0:
            features.append(f'Ch. {int(i/no_conditions)+1} Disrupted')
        if i%no_conditions == 1:
            features.append(f'Ch. {int(i/no_conditions)+1} Jammed')
        if i%no_conditions == 2:
            features.append(f'Ch. {int(i/no_conditions)+1} Successful Tx')
        if i%no_conditions == 3:
            features.append(f'Ch. {int(i/no_conditions)+1} Idle')
        if i%no_conditions == 4:
            features.append(f'Ch. {int(i/no_conditions)+1} Disrupted/Jammed')
        if i%no_conditions == 5:
            features.append(f'Ch. {int(i/no_conditions)+1} Disrupted/Successful')
        if i%no_conditions == 6:
            features.append(f'Ch. {int(i/no_conditions)+1} Disrupted/Idle')

    fig = plt.figure(1)
    ax = fig.add_axes([.1, .3, .8, .7])
    ax.bar([m for m in range(len(imps))], imps)
    temp = 0
    ax.set_xticks(np.arange(len(features)))
    ax.set_xticklabels(features)
    for tick in ax.get_xticklabels():
        tick.set_rotation(270)
        tick.set_visible(True)
        #print(features[temp])
        #tick.set_text(features[temp])
        temp += 1
    #plt.show()
    fig.savefig('report/feature_importance_1.jpeg')

    fig = plt.figure(2)
    ax = fig.add_axes([.1, .1, .8, .8])
    temp = []
    for i in range(A):
        temp.append(sum(imps[i*no_conditions:i*no_conditions+no_conditions]))
    ax.bar([m for m in range(len(temp))], temp)
    dfImps = pd.DataFrame(temp)
    dfImps.to_csv(f'report/{scenario}_importances.csv', index=False)
    temp = 0
    ax.set_xticks(np.arange(A))
    ax.set_xticklabels(['Channel 1', 'Channel 2', 'Channel 3', 'Channel 4', 'Channel 5', 'Channel 6', 'Channel 7', 'Channel 8', 'Channel 9'])
    for tick in ax.get_xticklabels():
        tick.set_visible(True)
        #print(features[temp])
        #tick.set_text(features[temp])
        temp += 1
    #plt.show()
    fig.savefig('report/feature_importance_2.jpeg')

    #draw_tree(mdl, features, precision=3)
    #tree_text = tree.export_text(model2)
    #print(tree_text)
    #fig = plt.figure(figsize=(180,60)) # , dpi=200, figsize=(100,20)
    #tree.plot_tree(model, feature_names=features, filled=True, rounded=True, fontsize=50, node_ids=True, proportion=True, precision=2) #, fontsize=40
    #fig.savefig('decision_tree.png')
    #plt.show()
    
    tree.export_graphviz(model, feature_names=features, filled=True, out_file=f"treeStructures/DT_{scenario}.dot", rounded=True, node_ids=True, proportion=True)
    
    del model