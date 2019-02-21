# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 17:00:17 2019

@author: arian
"""

import tictactoe_etu as tictac
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def N_run(j1, j2, N = 200):
    """
    Simule N parties d'un jeu du Morphion et retourne 3 tableaux avec les gains
    du premier joueur, du deuxième joueur et de matchs muls en N parties.
    """
    vic_1 = np.zeros(N)
    vic_2 = np.zeros(N)
    nul = np.zeros(N)  
    for i in range(N):
        state = tictac.MorpionState()
        jeu = tictac.Jeu(state, j1, j2)
        victoire, _ = jeu.run()
        if (victoire == 0):
            nul[i] = nul[i - 1] + 1
            vic_1[i] = vic_1[i - 1]
            vic_2[i] = vic_2[i - 1]
        if (victoire == -1):
            nul[i] = nul[i - 1]
            vic_1[i] = vic_1[i - 1]
            vic_2[i] = vic_2[i - 1] + 1
        if (victoire == 1):
            nul[i] = nul[i - 1]
            vic_1[i] = vic_1[i - 1] + 1
            vic_2[i] = vic_2[i - 1]     
    return vic_1, vic_2, nul


def graphe_victoires(j1, j2, N = 200):
    """
    Cree un graphe d'evolution des gains des 2 joueurs (et matchs muls) en N parties du jeu du Morpion.
    """    
    vic_1, vic_2, nul = N_run(j1, j2, N)
    T = np.arange(N)
    fig, ax = plt.subplots()
    ax.grid(True)
    ax.set_xlabel("N") 
    ax.plot(T, vic_1, label = 'victoire 1') 
    ax.plot(T, vic_2, label = 'victoire 2') 
    ax.plot(T, nul, label = 'nul') 
    ax.legend(loc = "upper left")
    

def graphe_vic_moy(j1, j2, N = 200):
    """
    Cree un graphe d’évolution de la moyenne du nombre de partie gagnée du premier joueur,
    deuxième joueur et des matchs nuls en N parties du jeu du Morpion.
    """    
    vic_1, vic_2, nul = N_run(j1, j2, N)
    T = np.arange(N) + 1
    vic_1 = vic_1 / T
    vic_2 = vic_2 / T   
    nul= nul / T
    
    fig, ax = plt.subplots()
    ax.grid(True)
    ax.set_xlabel("N") 
    ax.plot(T, vic_1, label = 'victoire 1') 
    ax.plot(T, vic_2, label = 'victoire 2') 
    ax.plot(T, nul, label = 'nul') 
    ax.legend(loc = "upper left")



if __name__ == "__main__": 
    #j1 = tictac.AgentAlea()
    #j1 = tictac.AgentMC()
    j1 = tictac.AgentMTTS()
    
    j2 = tictac.AgentAlea()
    #j2 = tictac.AgentMC()
    #j2 = tictac.AgentMTTS()

    N = 100
    #graphe_vic_moy(j1, j2, N)


    state = tictac.MorpionState()
    jeu = tictac.Jeu(state, j2, j1)
    victoire, log = jeu.run(draw=True, pause = 2)
    print(victoire)