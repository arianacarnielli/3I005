# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 17:00:17 2019

@author: arian
"""

import tictactoe_etu as tictac
import numpy as np
import matplotlib.pyplot as plt
import time

#==============================================================================
# L'implementátion des agents et du jeu Puissance 4 se trouvent dans le fichier
# tictactoe_etu qui contient aussi le code fait par le prof.
#==============================================================================


def N_run(j1, j2, N = 200, jeuState = tictac.MorpionState):
    """
    Simule N parties d'un jeu d'un jeu et retourne 3 tableaux avec les gains
    du premier joueur, du deuxième joueur et de matchs muls en N parties. JeuState identifie
    quel jeu on simule (Morpion ou Puissance 4).
    """
    vic_1 = np.zeros(N)
    vic_2 = np.zeros(N)
    nul = np.zeros(N)  
    for i in range(N):
        state = jeuState()
        jeu = tictac.Jeu(state, j1, j2)
        victoire, _ = jeu.run()
        nul[i] = nul[i - 1] + (victoire == 0)
        vic_1[i] = vic_1[i - 1] + (victoire == 1)
        vic_2[i] = vic_2[i - 1] + (victoire == -1)
    return vic_1, vic_2, nul


def graphe_victoires(j1, j2, N = 200):
    """
    Cree un graphe d'evolution des gains des 2 joueurs (et matchs muls) en N parties d'un jeu.
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
    deuxième joueur et des matchs nuls en N parties d'un jeu.
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
    #j1 = tictac.AgentMCTS()
    
    #j2 = tictac.AgentAlea()
    #j2 = tictac.AgentMC()
    #j2 = tictac.AgentMCTS()

    #N = 100
    #graphe_vic_moy(j1, j2, N)
    
    start = time.clock()
    #Pour des résultats plus précises, agumenter N. Fait attention, N trop grand peut prend
    #beaucoup de temps pour ternimer.
    N = 10
    tab_results = np.empty((9, 5))
    # Chaque ligne de tab_results représente une configuration j1, j2 différente.
    # Les deux premières colonnes donnent les indices des joueurs :
        # 0 : Alea
        # 1 : MC
        # 2 : MCTS
    # Les trois colonnes suivantes donnent, dans l'ordre, la proportion de victoires
    # du joueur 1, la proportion de matchs nuls et la proportion de victoires
    # du joueur 2
    k = 0
    for i, j1 in [(0, tictac.AgentAlea()), (1, tictac.AgentMC()), (2, tictac.AgentMCTS())]:
        for j, j2 in [(0, tictac.AgentAlea()), (1, tictac.AgentMC()), (2, tictac.AgentMCTS())]:
            print("i =",i, ", j =",j)
            vic_1, vic_2, nul = N_run(j1, j2, N, tictac.Puissance4State)
            tab_results[k] = [i, j, vic_1[-1]/N, nul[-1]/N, vic_2[-1]/N]
            k += 1
            print(time.clock() - start)


    #state = tictac.Puissance4State()
    #jeu = tictac.Jeu(state, j2, j1)
    #print(state.get_actions)
    #victoire, log = jeu.run(draw=True, pause = 1)
    #print(victoire)