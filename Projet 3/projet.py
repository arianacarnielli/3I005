#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 16:48:04 2019

@author: 3525837
"""

import Projet_Bioinfo as pb
import numpy as np
import math
import matplotlib.pyplot as plt

#==============================================================================
# Préliminaires : données et lecture des fichiers
#==============================================================================

def logproba(liste_entiers, m):
    """
    Calcule la log-probabilité de la séquence liste_entiers étant donné les
    fréquences de lettres dans m.
    
    m : tuple de taille 4 tel que m[i] contient la fréquence du nucléotide
        représenté par i.
    liste_entiers : liste d'entiers de 0 à 3.
    """
    res = 0
    for nucleo in liste_entiers:
        res += math.log(m[nucleo])
    return res

def logprobafast(liste_count, m):
    """
    Calcule la log-probabilité d'une séquence de nucléotides étant donné les
    fréquences de lettres dans m.
    
    m : tuple de taille 4 tel que m[i] contient la fréquence du nucléotide
        représenté par i.
    liste_count : liste de taille 4 contenant la quantité de fois que chaque
                  nucléotide apparaît dans la séquence.
    """
    res = 0
    for i in range(len(liste_count)):
        res += (math.log(m[i]) * liste_count[i])
    return res

#==============================================================================
# Annotation des régions promoteurs
#==============================================================================

#------------------------------------------------------------------------------
# Description Empirique, préliminaires
#------------------------------------------------------------------------------

def code(m, k):
    """
    Calcule, à partir d'un mot m de taille k, son indice dans le tableau
    ordonné lexicogaphiquement.
    
    m : liste de chiffres entre 0 et 3.
    k : len(m).
    """
    res = m[0]
    for i in range(1, k):
        res = 4*res + m[i]
    return res

#    Implémentation alternative avec complexité quadratique en nombre de
#    multiplications :
#    res = 0
#    puissance = k - 1
#    for i in range (len(m)):
#        res += pow(4, puissance - i) * m[i]
#    return res
    
def inverse(i, k):
    """
    Calcule, à partir d'un indice i et une taille k, le mot de taille k à
    l'indice i dans le tableau ordonné lexicogaphiquement.
    
    i : entier entre 0 (inclus) et 4**k (exclu).
    k : entier.
    """
    res = []
    div = i
    for j in range(k):
        div, reste = divmod(div, 4)
        res.append(reste)
    res.reverse()
    return res

#    Implémentation alternative avec complexité quadratique en nombre de
#    multiplications et divisions :        
#    res = []
#    div = i
#    for j in range(k - 1, -1, -1):
#        res.append(div // pow(4, j))
#        div = div % pow(4, j)
#    return res

def comptage(seq, k):
    """
    Compte le nombre d'occurences pour tous les mots de taille k dans la
    séquence seq.
    
    seq : séquence de nucléotides représentée par chaine de caractères ou
          par liste d'entiers entre 0 et 3.
    k : taille des mots à considérer.
    
    Renvoie un dictionnaire indexé par les mots de taille k avec le nombre
    d'occurences de ce mot dans la séquence. Des mots n'apparaissant pas dans
    la séquence ne sont pas inclus dans le dictionnaire.
    
    Si seq est une chaine de caractères, les clés du dictionnaire sont aussi
    des chaines de caractères représentant les mots de taille k. Sinon, elles
    sont les résultats de l'application de la fonction code aux séquences
    d'entiers représentant les mots.
    """
    dico = {}
    if isinstance(seq, str):
        for i in range(len(seq) - k + 1):
            temp = seq[i: i + k]
            if temp in dico:
                dico[temp] += 1
            else:
                dico[temp] = 1
    else:
        for i in range(len(seq) - k + 1):
            temp = seq[i: i + k]
            code_temp = code(temp, k)
            if code_temp in dico:
                dico[code_temp] += 1
            else:
                dico[code_temp] = 1
    return dico
    
def comptage_attendu(freq, k, l):
    """
    À partir des fréquences des lettres dans le génome donnée dans le tableau
    freq, calcule l'espérence du nombre d'occurences de chaque mot de longueur
    k dans une séquence de longueur l.
    
    freq : tuple de taille 4 tel que freq[i] contient la fréquence du
           nucléotide représenté par i.
    k : taille des mots à considérer.
    l : taille de la séquence à considérer.
    
    Renvoie un tableau tab tel que tab[i] contient l'espérence du nombre
    d'occurences du mot encodé par i.
    """
    tab = []
    for i in range(pow(4, k)):
        seq = inverse(i,k)
        tab.append(math.exp(logproba(seq, freq)) * (l - (k - 1)))
    return tab
        
def graphe_occurrences(freq, k, dict_sequences, filename = None):
    """
    Trace et enregistre le graphe avec les nombres d'occurences attendus et
    observés pour des mots de taille k.
    
    freq : tuple de taille 4 tel que freq[i] contient la fréquence du
           nucléotide représenté par i. Utilisé pour le calcul du nombre
           d'occurrences attendu de chaque mot.
    k : taille des mots à considérer.
    dict_sequences : dictionnaire avec les séquences pour les nombres
                     d'occurrences observées. Les clés doivent être des
                     chaînes de caractères décrivant les séquences.
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.grid(True)
    ax.set_axisbelow(True)
    
    minVal = float("inf")
    maxVal = -float("inf")
    
    for seq_name in dict_sequences:
        sequence = dict_sequences[seq_name]
        l = len(sequence)
        x = comptage_attendu(freq, k, l)
        y = comptage(sequence, k) # Dictionnaire. On le transforme en liste.
        y = [y[i] if i in y else 0 for i in range(4**k)]
        minVal = min(min(x), min(y), minVal)
        maxVal = max(max(x), max(y), maxVal)
        ax.scatter(x, y, label=seq_name, zorder=3)
    
    ax.plot([minVal, maxVal], [minVal, maxVal], color="#999999", zorder=2)
    ax.set_aspect("equal")
    ax.set_xlabel("Nombre attendu")
    ax.set_ylabel("Nombre observé")
    ax.set_title("Nombre d'occurrences pour des mots de taille {:d}".format(k))
    ax.legend(loc="lower right")
    if filename is not None:
        fig.savefig("Rapport/Figures/"+filename)
    plt.show()
    
#------------------------------------------------------------------------------
# Simulation de séquences aléatoires
#------------------------------------------------------------------------------
def simule_sequence(lg, m):
    """
    Génère une séquence aléatoire de longueur lg d’une composition donnée m 
    (proportion de A, C, G et T).
    lg : taille de la sequence à simuler.
    m : tuple de taille 4 tel que m[i] contient la fréquence du
        nucléotide représenté par i.
    """
    return np.random.choice(4, lg, p = m)

def proba_empirique(mot, lg, m, N):
    """
    Étant donné un mot, une longueur de séquence lg, les probabilités de chaque
    lettre et un entier N, estime la loi de probabilité du nombre d'occurences
    de mot dans des séquences aléatoires de longueur lg à partir de N tirages
    de séquences aléatoires.
    
    mot : mot codé comme liste de nombres de 0 à 3.
    lg : taille de la sequence à simuler.
    m : tuple de taille 4 tel que m[i] contient la fréquence du
        nucléotide représenté par i.
    N : nombre de simulations à faire.
    
    Renvoie un dictionnaire proba tel que proba[i] donne la probabilité estimée
    que le mot m apparaisse exactement i fois dans une séquence de longueur lg.
    Si la probabilité estimée vaut 0, i n'apparait pas dans les clés du
    dictionnaire.
    """
    proba = {}
    k = len(mot)
    code_mot = code(mot, k)
    for i in range(N):
        seq = simule_sequence(lg, m)
        compt = comptage(seq, k)
        if code_mot in compt:
            cpt = compt[code_mot]
        else:
            cpt = 0
        if cpt in proba:
            proba[cpt] += 1
        else:
            proba[cpt] = 1
    return {cpt : proba[cpt]/N for cpt in proba}

#------------------------------------------------------------------------------
# Modèles de dinucléotides et trinucléotides
#------------------------------------------------------------------------------

def estim_M(seq):
    """
    Estime la matrice M d'une chaine de Markov à partir d'une séquence produite
    par cette chaine.
    
    seq : séquence de nucléotides représentée par liste d'entiers entre 0 et 3.
    """
    cpt = comptage(seq, 2)
    M = np.zeros((4, 4))
    for cle in cpt:
        lettres = inverse(cle, 2)
        M[lettres[0], lettres[1]] = cpt[cle]
    for i in range(4):
        M[i, :] /= M[i, :].sum()
    return M

def simule_Markov(m, M, lg):
    """
    Simule une séquence de longueur lg à partir de la chaîne de Markov définie
    par la matrice de transition M avec la probalité initiale m.
    
    m : tuple de taille 4 tel que m[i] contient la fréquence du
        nucléotide représenté par i.
    M : matrice de transition de la chaîne de Markov.
    lg : longueur de la séquence à simuler.
    """
    seq = np.empty(lg, dtype=int)
    seq[0] = np.random.choice(4, p = m)
    for i in range(1, lg):
        seq[i] = np.random.choice(4, p = M[seq[i-1], :])
    return seq

def logproba_mot_Markov(mot, m, M):
    """
    Calcule le log de la probabilité que le mot passé en argument soit produit
    par la chaîne de Markov de matrice de transition M et probabilité
    invariante m.
    
    mot : mot codé comme liste de nombres de 0 à 3.
    m : probabilité invariante de la chaîne de Markov.
    M : matrice de transition de la chaîne de Markov.
    """
    res = math.log(m[mot[0]])
    for i in range(1, len(mot)):
        res += math.log(M[mot[i-1], mot[i]])
    return res
    
def comptage_attendu_Markov(m, M, k, l):
    """
    À partir de la chaîne de Markov déterminée par la matrice de transition M
    et de probabilité invariante m, calcule l'espérence du nombre d'occurences
    de chaque mot de longueur k dans une séquence de longueur l.
    
    m : probabilité invariante de la chaîne de Markov.
    M : matrice de transition de la chaîne de Markov.
    k : taille des mots à considérer.
    l : taille de la séquence à considérer.
    
    Renvoie un tableau tab tel que tab[i] contient l'espérence du nombre
    d'occurences du mot encodé par i.
    """
    tab = []
    for i in range(pow(4, k)):
        seq = inverse(i,k)
        tab.append(math.exp(logproba_mot_Markov(seq, m, M)) * (l - (k - 1)))
    return tab

def graphe_occurrences_Markov(m, M, k, dict_sequences, filename = None):
    """
    Trace et enregistre le graphe avec les nombres d'occurences attendus et
    observés pour des mots de taille k. Le calcul du nombre attendu est fait
    en supposant un modèle par chaîne de Markov d'ordre 1.
    
    m : probabilité invariante de la chaîne de Markov.
    M : matrice de transition de la chaîne de Markov.
    k : taille des mots à considérer.
    dict_sequences : dictionnaire avec les séquences pour les nombres
                     d'occurrences observées. Les clés doivent être des
                     chaînes de caractères décrivant les séquences.
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.grid(True)
    ax.set_axisbelow(True)
    
    minVal = float("inf")
    maxVal = -float("inf")
    
    for seq_name in dict_sequences:
        sequence = dict_sequences[seq_name]
        l = len(sequence)
        x = comptage_attendu_Markov(m, M, k, l)
        y = comptage(sequence, k) # Dictionnaire. On le transforme en liste.
        y = [y[i] if i in y else 0 for i in range(4**k)]
        minVal = min(min(x), min(y), minVal)
        maxVal = max(max(x), max(y), maxVal)
        ax.scatter(x, y, label=seq_name, zorder=3)
    
    ax.plot([minVal, maxVal], [minVal, maxVal], color="#999999", zorder=2)
    ax.set_aspect("equal")
    ax.set_xlabel("Nombre attendu")
    ax.set_ylabel("Nombre observé")
    ax.set_title("Nombre d'occurrences pour des mots de taille {:d}".format(k))
    ax.legend(loc="lower right")
    if filename is not None:
        fig.savefig("Rapport/Figures/"+filename)
    plt.show()
    
#------------------------------------------------------------------------------
# Probabilités de mots
#------------------------------------------------------------------------------