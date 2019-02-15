# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 17:00:17 2019

@author: arian
"""

import tictactoe_etu as tictac
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

j1 = tictac.AgentAlea()
j2 =tictac.AgentAlea()

N = 200

vic_1 = np.zeros(N)
vic_2 = np.zeros(N)
pers = np.zeros(N)

for i in range(N):
    state = tictac.MorpionState()
    jeu = tictac.Jeu(state, j1, j2)
    victoire, log = jeu.run()
    if (victoire == 0):
        pers[i] = pers[i - 1] + 1
        vic_1[i] = vic_1[i - 1]
        vic_2[i] = vic_2[i - 1]
    if (victoire == -1):
        pers[i] = pers[i - 1]
        vic_1[i] = vic_1[i - 1]
        vic_2[i] = vic_2[i - 1] + 1
    if (victoire == 1):
        pers[i] = pers[i - 1]
        vic_1[i] = vic_1[i - 1] + 1
        vic_2[i] = vic_2[i - 1] 

    
T = np.arange(N)
fig, ax = plt.subplots()
ax.grid(True)
ax.set_xlabel("N") 
ax.plot(T, vic_1, label = 'victoire 1') 
ax.plot(T, vic_2, label = 'victoire 2') 
ax.plot(T, pers, label = 'nulle') 
plt.legend(loc = "upper left")
    

if __name__ == "__main__": 
    state = tictac.MorpionState()
    
    jeu = tictac.Jeu(state, j1, j2)
    
    #victoire, log = jeu.run(draw=True)
    
    #print(victoire)