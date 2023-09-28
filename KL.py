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
from math import log2

def KL(p, q):
    summ = 0.0
    for i in range(len(q)):
        if p[i] == 0 and q[i] == 0:
            pass # summ += 0.0
        elif p[i] == 0:
            pass #summ += 0.0000000000000001 * log2(0.0000000000000001 / q[i])
        elif q[i] == 0:
            summ += p[i] * log2(p[i] / 0.0000000000000001)
        else:
            summ += p[i] * log2(p[i] / q[i])
    return summ

def getArray(add):
    try:
        dfArr = pd.read_csv(add)
        arr = dfArr.to_numpy()
        return arr
    except:
        return []

scenario1 = input('Scenario 1? ')
scenario2 = input('Scenario 2? ')

imps1 = getArray(f'report/{scenario1}_importances.csv')
imps2 = getArray(f'report/{scenario2}_importances.csv')
uni = [1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9]
imps1theory = [.05, .05, .05, .05, .1, .05, .4, .1, .15]
imps2theory = [.15, .1, .25, .1, .05, .05, .1, .15, .05]
imps3theory = [.15, .15, .1, .1, .1, .1, .1, .1, .1]#[.01, .06, .06, .2, .2, .2, .2, .06, .01]
imps4theory = [.016, .23, .016, .23, .016, .23, .016, .23, .016]#[.01, .01, .01, .55, .01, .2, .01, .19, .01]
impsDtheory = [1/6, 1/6, 1/12, 1/12, 1/12, 1/6, 1/12, 1/12, 1/12]
print(f'KL({scenario1}, {scenario2}) = {KL(imps1, imps2)}')
print(f'KL({scenario2}, {scenario1}) = {KL(imps2, imps1)} avg = {( KL(imps1, imps2) + KL(imps2, imps1) ) / 2}')
print('uniform')
print(f'KL({scenario1}, uniform) = {KL(imps1, uni)}')
print(f'KL(uniform, {scenario1}) = {KL(uni, imps1)} avg = {( KL(imps1, uni) + KL(uni, imps1) ) / 2}')
print('absolutely theory')
print(f'(1, uniform) = {KL(imps1theory, uni)}')
print(f'KL(uniform, 1) = {KL(uni, imps1theory)} avg = {( KL(imps1theory, uni) + KL(uni, imps1theory) ) / 2}')
print(f'(2, uniform) = {KL(imps2theory, uni)}')
print(f'KL(uniform, 2) = {KL(uni, imps2theory)} avg = {( KL(imps2theory, uni) + KL(uni, imps2theory) ) / 2}')
print(f'(3, uniform) = {KL(imps3theory, uni)}')
print(f'KL(uniform, 3) = {KL(uni, imps3theory)} avg = {( KL(imps3theory, uni) + KL(uni, imps3theory) ) / 2}')
print(f'(4, uniform) = {KL(imps4theory, uni)}')
print(f'KL(uniform, 4) = {KL(uni, imps4theory)} avg = {( KL(imps4theory, uni) + KL(uni, imps4theory) ) / 2}')
print(f'(dynamic, uniform) = {KL(impsDtheory, uni)}')
print(f'KL(uniform, dynamic) = {KL(uni, impsDtheory)} avg = {( KL(impsDtheory, uni) + KL(uni, impsDtheory) ) / 2}')