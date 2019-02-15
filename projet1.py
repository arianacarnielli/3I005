#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 15:57:57 2019

@author: 3302858
"""

import numpy as np 
import matplotlib.pyplot as plt

#==============================================================================
#Common
#==============================================================================

def jouer(machine, action):
    """
    Prend en argument une machine sous forme de liste et un entier, l’action/levier 
    choisi, et rend le gain binaire correspondant au coup joué.
    """
    res=np.random.random() 
    dist=machine[action]
    if (res < dist): 
        return 1
    return 0

def mise_a_jour(mu, Na, choix, res):
    """
    Mise a jour des tableaux mu et Na.
    mu = tableau des probas estimées pour chaque machine
    Na = tableau qui garde la quantité de fois qu'on a joué a chaque machine[i] 
    """
    mu[choix] = ((mu[choix] * Na[choix]) + res) / (Na[choix]+1)
    Na[choix] += 1
    
def cree_machine(quant):
    """
    Cree un vecteur machine de façon aleatoire, avec n leviers.
    """
    return np.random.rand(quant)   
    
#==============================================================================
#Algorithmes
#Tous les algorithmes prennent mu et Na comme arguments.
#  mu = tableau des probas estimées pour chaque machine
#  Na = tableau qui garde la quantité de fois qu'on a joué a chaque machine[i]
#==============================================================================

def algo_alea(mu, Na):
    """
    L'algorithme aléatoire.
    """
    return np.random.randint(0, Na.size)

def algo_glouton(mu, Na, pphase = 10):
    """
    L’algorithme "greedy".
    Premier phase = Na[i] < 10 par defaut
    """
    i = 0
    while i < Na.size:
        if Na[i] < pphase:
            return i
        else:
            i+= 1
    return np.argmax(mu)
    
def algo_glouton_e(mu, Na, e = 0.1): 
    """
    L’algorithme e-greedy.
    e = 0.1 par defaut
    """
    alea = np.random.random() 
    if alea < e :
        return algo_alea(mu, Na)
    else :
        return algo_glouton(mu, Na)
    
def algo_UCB(mu, Na):
    """
    L'algorithme Upper Confidence Bound.
    """
    i = 0
    while i < Na.size:
        if Na[i] < 1:
            return i
        else:
            i+= 1
    t = Na.sum()
    return np.argmax(mu + np.sqrt(2*np.log(t)/Na))
    
def run(machine, algo, T):
    """
    Simule T coups de jeu, avec l'ensemble de machines machine et l'algorithme
    passe en paramètre. 
    Retourne mu, Na, res_temps
    mu = tableau des probas estimées pour chaque machine a la fin
    Na = tableau qui garde la quantité de fois qu'on a joué a chaque machine[i] a la fin
    res_temps = tableau avec les resultats partiels a chaque t donné     
    """
    mu = np.zeros(machine.size) #tableau des probas estimées pour chaque machine 
    Na = np.zeros(machine.size, dtype = np.int32) #nb fois qu'on a joué machine(i) 
    res_temps = np.zeros(T, dtype = np.int32) 
    for t in range(T):
        choix= algo(mu, Na)
        res = jouer(machine, choix) 
        res_temps[t] = res_temps[t - 1] + res
        mise_a_jour(mu, Na, choix, res)
    return mu, Na, res_temps

#==============================================================================
#Experiences
#Quelques fontions pour calculer le regret, le gain optimal, etc.
#Fonctions pour creer des graphes.
#==============================================================================
def gain_opt_total(machine, T):
    """
    Calcule le gain maximal à un temps T.
    """
    return (np.amax(machine) * T)

def gain_opt(machine, T):
     """
     Calcule le gain maximal a toutes les instances entre 0 e T.
     Retourne un tableau de ces gains.
     """
     res = (np.arange(T)+1)
     return res * np.amax(machine)
    
def regret(machine, T, res_temps):
    """
    Calcule l'evolution du regret par rapport au temps.
    Renvoie un tableau des regrets pour chaque t entre 0 et T.   
    """
    opt = gain_opt(machine, T)
    return opt - res_temps

def moyenne_run(machine, algo, T, n = 100):
    """
    Appele la fonction run n fois appliqué sur les autres arguments.
    Calcule res_temps moye et mu_moyen et renvoie mu_moyen, Na et res_temps_moyen.
    """
    mu_moyen, Na, res_temps_moyen =  run(machine, algo, T) 
    for i in range(1, n):
        mu, Na, res = run(machine, algo, T)
        mu_moyen = mu_moyen + mu
        res_temps_moyen = res_temps_moyen + res
    return mu_moyen/n, Na, res_temps_moyen/n  

def graphe_regret_temps(opt, rgt, res):
    """
    Cree un graphe d'evolution du gain reel, gain optimal e regret.
    Les 3 resultats doivent etre calcules auparavant.
    """
    T = np.arange(opt.size)
    fig, ax = plt.subplots()
    ax.grid(True)
    plt.xlabel("T")
   
    ax.plot(T, rgt, label = 'regret') 
    ax.plot(T, opt, label = 'optimal')
    ax.plot(T, res, label = 'resultat obtenu')
    plt.legend(loc = "upper left")
    
def graphe_regrets(rgt_alea, rgt_glou, rgt_glou_e, rgt_ucb):
    """
    Cree un graphe d'evolution des regrets des 4 algorithmes.
    Les regrets doivent etre calcules auparavant.
    """
    T = np.arange(rgt_alea.size)
    fig, ax = plt.subplots()
    ax.grid(True)
    plt.xlabel("T")
    plt.ylabel("Regret")
   
    ax.plot(T, rgt_alea, label = 'aléatoire') 
    ax.plot(T, rgt_glou, label = 'glouton')
    ax.plot(T, rgt_glou_e, label = 'e-glouton')
    ax.plot(T, rgt_ucb, label = 'UCB')
    ax.legend(loc = "upper left")
    plt.title("regrets des 4 algorithmes par rapport à T")
    
def graphe_gains(res_alea, res_glou, res_glou_e, res_ucb):
    """
    Cree un graphe d'evolution des gains des 4 algorithmes.
    Les gains doivent etre calcules auparavant.
    """
    T = np.arange(res_alea.size)
    fig, ax = plt.subplots()
    ax.grid(True)
    plt.xlabel("T")
    plt.ylabel("Gains")
   
    ax.plot(T, res_alea, label = 'aléatoire') 
    ax.plot(T, res_glou, label = 'glouton')
    ax.plot(T, res_glou_e, label = 'e-glouton')
    ax.plot(T, res_ucb, label = 'UCB')
    ax.legend(loc = "upper left")
    plt.title("Gains des 4 algorithmes par rapport à T")

#==============================================================================
#Testes  
#==============================================================================

if __name__ == "__main__":  
    machine = cree_machine(5)
    T = 100
    #mu, Na, res = run(machine, algo_alea, T)
    #mu, Na, res = run(machine, algo_glouton, T)
    #mu, Na, res = run(machine, algo_glouton_e, T)
    #mu, Na, res = run(machine, algo_UCB, T)
    
    mu_alea, Na, res_alea = moyenne_run(machine, algo_alea, T)
    mu_glou, Na, res_glou = moyenne_run(machine, algo_glouton, T)
    mu_glou_e, Na, res_glou_e = moyenne_run(machine, algo_glouton_e, T)
    mu_ucb, Na, res_ucb = moyenne_run(machine, algo_UCB, T)
    
    rgt_alea = regret(machine, T, res_alea)
    rgt_glou = regret(machine, T, res_glou)
    rgt_glou_e = regret(machine, T, res_glou_e)
    rgt_ucb = regret(machine, T, res_ucb)
    
    #print("mu = ", mu)
    #print("\n")
    #print("Na = ", Na)
    #print("\n")
    #print("res = ", res)
    #print("\n")
    opt_total = gain_opt_total(machine, T)
    print("gain optimal total = ", opt_total)
    print("\n")
    opt = gain_opt(machine, T)
    print("gain optimal = ", opt)
    print("\n")
    #rgt = regret(machine, T, res)
    #print("regret = ", rgt)
    #print("\n")
    #rgt_total = opt_total - res[-1]
    #print("regret_total = ", rgt_total)
    graphe_regret_temps(opt, rgt_glou, res_glou)
    graphe_regrets(rgt_alea, rgt_glou, rgt_glou_e, rgt_ucb)
    graphe_gains(res_alea, res_glou, res_glou_e, res_ucb)