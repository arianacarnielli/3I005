# -*- coding: utf-8 -*-
"""
TME 1 de 3I005
"""
import random as rd
import matplotlib.pyplot as plt

def randList(n, a=0, b=10):
    return [rd.randint(a, b) for i in range(n)]

def moyenne(l):
    return sum(l)/len(l) 

def histo(l):
    dico = {}
    for i in l:
        if i in dico:
            dico[i] += 1
        else:
            dico[i] = 1
    return dico 

def histo_trie(l):
   dico = histo(l)
   l = [(dico[k], k)for k in dico]
   return sorted(l)

def paquet():
    l=[(num, coul) for coul in "CKPT" for num in range(1, 14)]
    rd.shuffle(l)
    return l 

def meme_position(p,q):
    return [i for i in range(len(p)) if p[i] == q[i]]
            
def stat_meme_position(n = 10000):
    l=[]
    cpt = 0
    for i in range(n):
        cpt+= len(meme_position(paquet(), paquet()))
        l.append(cpt/(i+1))
    return l

def plot_stat_meme_position(n = 10000):
    liste_x = range(n)
    liste_y = stat_meme_position(n)
    
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.set_xlim((0, n))
    ax.set_ylim((0, 2))
    ax.set_title('moyenne du nombre de cartes identiques à une même position en fonction du nombre d’expériences')
    ax.set_xlabel("quantité d'experiences")
    ax.set_ylabel('moyenne du nombre de cartes identiques')
    ax.plot(liste_x, liste_y)
    
    plt.show()
    
def plot_histo_positions(n = 10000):
    l =  []
    for i in range (n):
         l += meme_position(paquet(), paquet())
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.hist(l, 52, color='b',alpha=0.5, ec='black')

def de():
    return rd.randint(1, 6)

def proba_somme(k, n):
    l = []
    for i in range (n):
        somme = 0
        for j in range (k):
            somme+= de()
        l.append(somme)
    dico = histo(l)
    l = [(k, dico[k])for k in dico]
    for i in range(len(l)):
        l[i] = (l[i][0], (l[i][1]/n))
    return sorted(l)
        
def roulette(distribution):
    dist = sorted(distribution, key = lambda couple : couple[1])
    print(dist)
    cpt = 0
    res = rd.random()
    print(res)
    for ev, proba in dist:
        if (cpt <= res) and (res < (cpt + proba)):
            return ev
        else:
            cpt+= proba


if __name__ == "__main__":        

#    test = randList(5)
#    print(test)
#    print("\n")       
#    print(moyenne(test))
#    print("\n")  
#    x = randList(20)
#    print(x)
#    print("\n")
#    print(histo(x))
#    print("\n")
#    x = ["a", "b", "c", "d", "c", "e", "a", "a"]
#    print(histo_trie(x)) 
#    print(paquet())
#    q = paquet()
#    p = paquet()
#    print(q)
#    print("\n")
#    print(p)
#    print("\n")
#    print(meme_position(p, q))     
#    print(stat_meme_position())
#    plot_stat_meme_position()
#    plot_histo_positions()
#    print(de())   
#    print(proba_somme(2,1000))

    print(roulette([("P",0.6), ("OO", 0.1), ("F",0.3)]))