#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 16:55:58 2019

@author: 3302858
"""

import numpy as np 
import pandas as pd
from IPython.display import Image
import pydotplus
import utils

import matplotlib
import matplotlib.pyplot as plt  # for plotting
import seaborn as sns  # for making plots with seaborn

import math  # for math
import scipy.stats

#==============================================================================
# Question 1
#==============================================================================
def getPrior(df):
    """
    Calcule la probabilité a priori de la classe 1 ainsi que l'intervalle de
    confiance à 95% pour l'estimation de cette probabilité.
    
    :param df: Dataframe contenant les données. Doit contenir une colonne
               nommée "target" ne contenant que des 0 et 1.
    :type df: pandas dataframe
    :return: Dictionnaire contennant la moyenne et les extremités de l'intervalle
             de confiance. Clés 'estimation', 'min5pourcent', 'max5pourcent'.
    :rtype: Dictionnaire
    """
    dico =  {}
    moy = df["target"].mean()
    z = 1.96
    temp = z * math.sqrt((moy * (1 - moy))/df.shape[0])
    dico['estimation'] = moy
    dico['min5pourcent'] = moy - temp
    dico['max5pourcent'] = moy + temp
    
    return dico

#==============================================================================
# Question 2
#==============================================================================
class APrioriClassifier (utils.AbstractClassifier):
    """
    Estime très simplement la classe de chaque individu par la classe majoritaire.
    """
    def __init__(self):
        pass
    
    def estimClass(self, attrs):
        """
        à partir d'un dictionanire d'attributs, estime la classe 0 ou 1
        
        Pour ce APrioriClassifier, la classe vaut toujours 1.
        
        :param attrs: le  dictionnaire nom-valeur des attributs
        :return: la classe 0 ou 1 estimée
        """
        return 1    

    def statsOnDF(self, df):
        """
        à partir d'un pandas.dataframe, calcule les taux d'erreurs de classification et rend un dictionnaire.
        
        :param df:  le dataframe à tester
        :return: un dictionnaire incluant les VP,FP,VN,FN,précision et rappel
        
        VP : nombre d'individus avec target=1 et classe prévue=1
        VN : nombre d'individus avec target=0 et classe prévue=0
        FP : nombre d'individus avec target=0 et classe prévue=1
        FN : nombre d'individus avec target=1 et classe prévue=0
        Précision : combien de candidats sélectionnés sont pertinents (VP/(VP+FP))
        Rappel : combien d'éléments pertinents sont sélectionnés (VP/(VP+FN))
        """
        dico = {}
        dico["VP"] = 0
        dico["VN"] = 0
        dico["FP"] = 0
        dico["FN"] = 0
        for t in range(df.shape[0]):
            dic = utils.getNthDict(df, t)
            resultat = self.estimClass(dic)
            if dic["target"] == 1:
                if resultat == 1:
                    dico["VP"]+= 1
                else:
                    dico["FN"]+= 1
            else:
                if resultat == 1:
                    dico["FP"]+= 1
                else:
                    dico["VN"]+= 1
                    
        dico["Précision"] = dico["VP"]/(dico["VP"] + dico["FP"])
        dico["Rappel"] = dico["VP"]/ (dico["VP"] + dico["FN"])
        
        return dico
    
#==============================================================================
# Question 3
#==============================================================================
#------------------------------------------------------------------------------
# Question 3a
#------------------------------------------------------------------------------
def P2D_l(df, attr):
    """
    Calcul de la probabilité conditionnelle P(attribut | target).
    
    :param df: dataframe avec les données. Doit contenir une colonne nommée "target".
    :param attr: attribut à utiliser, nom d'une colonne du dataframe.
    :return: dictionnaire de dictionnaire dico. dico[t][a] contient P(attribut = a | target = t).
    :rtype: dictionnaire de dictionnaire.
    """
    list_cle = np.unique(df[attr].values) # Valeurs possibles de l'attribut.
    dico = {}
    # Les valeurs possibles pour target sont toujours 0 ou 1.
    dico[0] = dict.fromkeys(list_cle, 0)
    dico[1] = dict.fromkeys(list_cle, 0)
    for i, row in df.iterrows():
        dico[row["target"]][row[attr]]+=1
    
    taille0 = (df["target"] == 0).sum()
    taille1 = (df["target"] == 1).sum()
    
    for i in list_cle:
        dico[0][i] = dico[0][i]/taille0
        dico[1][i] = dico[1][i]/taille1
        
    return dico

def P2D_p(df, attr):
    """
    Calcul de la probabilité conditionnelle P(target | attribut).
    
    :param df: dataframe avec les données. Doit contenir une colonne nommée "target".
    :param attr: attribut à utiliser, nom d'une colonne du dataframe.
    :return: dictionnaire de dictionnaire dico. dico[a][t] contient P(target = t | attribut = a).
    :rtype: dictionnaire de dictionnaire.
    """
    list_cle = np.unique(df[attr].values) # Valeurs possibles de l'attribut
    dico = dict.fromkeys(list_cle)
    for cle in dico:
        dico[cle] = dict.fromkeys([0,1], 0) # Les valeurs possibles pour target sont toujours 0 ou 1.
    
    for i, row in df.iterrows():        
        dico[row[attr]][row["target"]]+=1
        
    for cle in dico:
        taille = (df[attr] == cle).sum()
        for i in range (2):
            dico[cle][i] = dico[cle][i] / taille
    return dico
    
#------------------------------------------------------------------------------
# Question 3b
#------------------------------------------------------------------------------    
class ML2DClassifier (APrioriClassifier):
    """
    Classifieur 2D par maximum de vraisemblance à partir d'une seule colonne du dataframe.
    """
    
    def __init__(self, df, attr):
        """
        Initialise le classifieur. Crée un dictionnaire de dictionnaires contenant
        les probabilités conditionnelles P(attribut | target).
        
        :param df: dataframe. Doit contenir une colonne appelée "target" ne contenant que 0 ou 1.
        :param attr: le nom d'une colonne du dataframe df.
        """
        self.attr = attr
        self.dico_p = P2D_l(df, attr)
        
    
    def estimClass(self, attrs):
        """
        à partir d'un dictionanire d'attributs, estime la classe 0 ou 1.        
        L'estimée est faite par maximum de vraisemblance à partir de dico_p.
        
        :param attrs: le  dictionnaire nom-valeur des attributs
        :return: la classe 0 ou 1 estimée
        """
        val = attrs[self.attr]
        #si la valeur de l'attribut n'existe pas dans l'enseble d'apprentissage,
        #alors sa probabilité conditionnelle vaut zero
        P = [0.0, 0.0]
        for t in [0, 1]:
            if val in self.dico_p[t]:
                P[t] = self.dico_p[t][val]
        if P[0] >= P[1]:
            return 0
        return 1
    
#------------------------------------------------------------------------------
# Question 3c
#------------------------------------------------------------------------------
class MAP2DClassifier (APrioriClassifier):
    """
    Classifieur 2D par maximum a posteriori à partir d'une seule colonne du dataframe.
    """
    
    def __init__(self, df, attr):
        """
        Initialise le classifieur. Crée un dictionnaire de dictionnaires contenant
        les probabilités conditionnelles P(target | attribut).
        
        :param df: dataframe. Doit contenir une colonne appelée "target" ne contenant que 0 ou 1.
        :param attr: le nom d'une colonne du dataframe df.
        """
        self.attr = attr
        self.dico_p = P2D_p(df, attr)
        
    
    def estimClass(self, attrs):
        """
        à partir d'un dictionanire d'attributs, estime la classe 0 ou 1.        
        L'estimée est faite par maximum a posteriori à partir de dico_p.
        
        :param attrs: le  dictionnaire nom-valeur des attributs
        :return: la classe 0 ou 1 estimée
        """
        val = attrs[self.attr] 
        if val in self.dico_p:
            if self.dico_p[val][0] >= self.dico_p[val][1]:
                return 0
            return 1
        else:
            #si la valeur de l'attribut n'existe pas dans l'enseble d'apprentissage,
            #alors sa probabilité conditionnelle n'est pas definie et on renvoie zero
            return 0
 
#==============================================================================
#Question 4
#==============================================================================
#------------------------------------------------------------------------------
# Question 4.1
#------------------------------------------------------------------------------
def nbParams(df, liste = None):
    """
    Affiche la taille mémoire de tables P(target|attr1,..,attrk) étant donné un
    dataframe df et la liste [target,attr1,...,attrl], en supposant qu'un float
    est représenté sur 8 octets. Pour cela la fonction utilise la fonction 
    auxiliaire octetToStr().
    
    :param df: Dataframe contenant les données. 
    :param liste: liste contenant les colonnes prises en considération pour le calcul. 
    """
    if liste is None:
        liste = list(df.columns) 
    taille = 8
    for col in liste:
        taille *= np.unique(df[col].values).size
    res = str(len(liste)) + " variable(s) : " + str(taille) + " octets"
    if taille >= 1024:
        res = res + " = " + octetToStr(taille)
    print (res)  

def octetToStr(taille):
    """
    Transforme l’entier taille en une chaîne de caractères qui donne sa représentation
    en nombre d’octets, ko, mo, go et to. 
    
    :param taille: le nombre à être transformé.
    """
    suffixe = ["o", "ko", "mo", "go", "to"]
    res = ""
    for suf in suffixe:
        if taille == 0:
            break  
        if suf == "to":
            res = " {:d}".format(taille) + suf + res
        else:
            res = " {:d}".format(taille % 1024) + suf + res
            taille //= 1024
    
    if res == "":
        res = " 0o"
    return res[1:]

#------------------------------------------------------------------------------
# Question 4.2
#------------------------------------------------------------------------------   
def nbParamsIndep(df):
    """
    Affiche la taille mémoire nécessaire pour représenter les tables de probabilité
    étant donné un dataframe, en supposant l'indépendance des variables et qu'un
    float est représenté sur 8 octets.Pour cela la fonction utilise la fonction 
    auxiliaire octetToStr().
    
    :param df: Dataframe contenant les données.  
    """
    taille = 0
    liste = list(df.columns) 
    
    for col in liste:
        taille += (np.unique(df[col].values).size * 8)
    
    res = str(len(liste)) + " variable(s) : " + str(taille) + " octets"
    if taille >= 1024:
        res = res + " = " + octetToStr(taille)
    print (res)
    
#==============================================================================
#Question 5
#==============================================================================
#------------------------------------------------------------------------------
# Question 5.3
#------------------------------------------------------------------------------    
def drawNaiveBayes(df, col):
    """
    Construit un graphe orienté représentant naïve Bayes.
    
    :param df: Dataframe contenant les données.  
    :param col: le nom de la colonne du Dataframe utilisée comme racine.
    """
    tab_col = list(df.columns.values)
    tab_col.remove(col)
    res = ""
    for enfant in tab_col:
        res = res + col + "->" + enfant + ";"
    return utils.drawGraph(res[:-1])    
    
def nbParamsNaiveBayes(df, col_pere, liste_col = None):
    """
    Affiche la taille mémoire de tables P(target), P(attr1|target),.., P(attrk|target) 
    étant donné un dataframe df, la colonne racine col_pere et la liste [target,attr1,...,attrk],
    en supposant qu'un float est représenté sur 8 octets. Pour cela la fonction 
    utilise la fonction auxiliaire octetToStr().
    
    :param df: Dataframe contenant les données. 
    :param col_pere: le nom de la colonne du Dataframe utilisée comme racine.
    :param liste_col: liste contenant les colonnes prises en considération pour le calcul. 
    """
    taille = np.unique(df[col_pere].values).size * 8
    
    if liste_col is None:
        liste_col = list(df.columns) 
        
    if liste_col != []:  
        liste_col.remove(col_pere)
    
    for col in liste_col:
        temp = (np.unique(df[col_pere].values).size * np.unique(df[col].values).size) * 8
        taille += temp
    
    res = str(len(liste_col)) + " variable(s) : " + str(taille) + " octets"
    if taille >= 1024:
        res = res + " = " + octetToStr(taille)
    print (res)
    
#------------------------------------------------------------------------------
# Question 5.4
#------------------------------------------------------------------------------      
class MLNaiveBayesClassifier(APrioriClassifier):
    """
    Classifieur par maximum de vraissemblance utilisant le modèle naïve Bayes. 
    """
    def __init__(self, df):
        """
        Initialise le classifieur. Crée un dictionnarie où la clé est le nom de
        chaque attribut et la valeur est un dictionnaire de dictionnaires contenant
        les probabilités conditionnelles P(attribut | target).
        
        :param df: dataframe. Doit contenir une colonne appelée "target" ne contenant que 0 ou 1.
        """
        self.dico_P2D_l = {}
        tab_col = list(df.columns.values)
        tab_col.remove("target")
        for attr in tab_col:
            self.dico_P2D_l[attr] = P2D_l(df, attr)
    
    def estimClass(self, attrs):
        """
        À partir d'un dictionanire d'attributs, estime la classe 0 ou 1.        
        L'estimée est faite par maximum de vraissemblance à partir de dico_res.
        
        :param attrs: le dictionnaire nom-valeur des attributs
        :return: la classe 0 ou 1 estimée
        """
        dico_res = self.estimProbas(attrs)
        if dico_res[0] >= dico_res[1]:
            return 0
        return 1
        
    def estimProbas(self, attrs):
        """
        Calcule la vraisemblance par naïve Bayes : P(attr1, ..., attrk | target).
        
        :param attrs: le dictionnaire nom-valeur des attributs
        """
        P_0 = 1
        P_1 = 1
        for attr_key in attrs:
            if attr_key != "target":
                dico_p = self.dico_P2D_l[attr_key]
                if attrs[attr_key] in dico_p[0]:
                    P_0 *= dico_p[0][attrs[attr_key]]
                    P_1 *= dico_p[1][attrs[attr_key]]
                else:
                    #si la valeur de l'attribut n'existe pas dans l'enseble d'apprentissage,
                    #alors sa probabilité conditionnelle vaut zero et on peut
                    #faire une sortie anticipée
                    return {0: 0.0, 1: 0.0}
        return {0: P_0, 1: P_1}

class MAPNaiveBayesClassifier(APrioriClassifier):
    """
    Classifieur par le maximum a posteriori en utilisant le modèle naïve Bayes. 
    """
    def __init__(self, df):
        """
        Initialise le classifieur. Crée un dictionnarie où la clé est le nom de
        chaque attribut et la valeur est un dictionnaire de dictionnaires contenant
        les probabilités conditionnelles P(attribut | target). Cree aussi un 
        dictionnaire avec les probabilités de target = 0 et target = 1.
        
        :param df: dataframe. Doit contenir une colonne appelée "target" ne contenant que 0 ou 1.
        """
        self.pTarget = {1: df["target"].mean()}
        self.pTarget[0] = 1 - self.pTarget[1] 
        self.dico_P2D_l = {}
        tab_col = list(df.columns.values)
        tab_col.remove("target")
        for attr in tab_col:
            self.dico_P2D_l[attr] = P2D_l(df, attr)
    
    def estimClass(self, attrs):
        """
        À partir d'un dictionanire d'attributs, estime la classe 0 ou 1.        
        L'estimée est faite par maximum à posteriori à partir de dico_res.
        
        :param attrs: le dictionnaire nom-valeur des attributs
        :return: la classe 0 ou 1 estimée
        """
        dico_res = self.estimProbas(attrs)
        if dico_res[0] >= dico_res[1]:
            return 0
        return 1
        

    def estimProbas(self, attrs):
        """
        Calcule la probabilité à posteriori par naïve Bayes : P(target | attr1, ..., attrk).
        
        :param attrs: le dictionnaire nom-valeur des attributs
        """
        P_0 = self.pTarget[0]
        P_1 = self.pTarget[1]
        for attr_key in attrs:
            if attr_key != "target":
                dico_p = self.dico_P2D_l[attr_key]
                if attrs[attr_key] in dico_p[0]:
                    P_0 *= dico_p[0][attrs[attr_key]]
                    P_1 *= dico_p[1][attrs[attr_key]]
                else:
                    #si la valeur de l'attribut n'existe pas dans l'enseble d'apprentissage,
                    #alors sa probabilité conditionnelle n'est pas definie et on peut
                    #faire une sortie anticipée
                    return {0: 0.0, 1: 0.0}
        P_0res = P_0 / (P_0 + P_1)
        P_1res = P_1 / (P_0 + P_1)
        return {0: P_0res, 1: P_1res}

#==============================================================================
# Question 6
#==============================================================================
def isIndepFromTarget(df,attr,x):
    """
    Vérifie si attr est indépendant de target au seuil de x%.
    
    :param df: dataframe. Doit contenir une colonne appelée "target" ne contenant que 0 ou 1.
    :param attr: le nom d'une colonne du dataframe df.
    :param x: seuil de confiance.
    """
    list_val = np.unique(df[attr].values) # Valeurs possibles de l'attribut.
    dico_val = {list_val[i]: i for i in range(list_val.size)} 
    #un dictionnaire associant chaque valeur a leur indice en list_val.
    
    mat_cont = np.zeros((2, list_val.size), dtype = int)
    
    for i, row in df.iterrows():
        j =  row[attr]
        mat_cont[row["target"], dico_val[j]]+= 1 
    
    _, p, _, _ = scipy.stats.chi2_contingency(mat_cont)
    return p > x
    
class ReducedMLNaiveBayesClassifier(APrioriClassifier):
    """
    Classifieur par maximum de vraissemblance utilisant le modèle naïve Bayes reduit. 
    """
    def __init__(self, df, x):
        """
        Initialise le classifieur. Crée un dictionnarie où la clé est le nom de
        chaque attribut et la valeur est un dictionnaire de dictionnaires contenant
        les probabilités conditionnelles P(attribut | target) où attribut et target
        ne sont pas indépendants.
        
        :param df: dataframe. Doit contenir une colonne appelée "target" ne contenant que 0 ou 1.
        :param x: seuil de confiance pour le test de indépendance.
        """
        self.dico_P2D_l = {}
        tab_col = list(df.columns.values)
        tab_col.remove("target")
        for attr in tab_col:
            if not isIndepFromTarget(df,attr,x):
                self.dico_P2D_l[attr] = P2D_l(df, attr)
    
    def estimClass(self, attrs):
        """
        À partir d'un dictionanire d'attributs, estime la classe 0 ou 1.        
        L'estimée est faite par maximum de vraissemblance à partir de dico_res.
        
        :param attrs: le dictionnaire nom-valeur des attributs
        :return: la classe 0 ou 1 estimée
        """
        dico_res = self.estimProbas(attrs)
        if dico_res[0] >= dico_res[1]:
            return 0
        return 1
        
    def estimProbas(self, attrs):
        """
        Calcule la vraisemblance par naïve Bayes : P(attr1, ..., attrk | target).
        
        :param attrs: le dictionnaire nom-valeur des attributs
        """
        P_0 = 1
        P_1 = 1
        for key in self.dico_P2D_l:
            dico_p = self.dico_P2D_l[key]
            if attrs[key] in dico_p[0]:
                P_0 *= dico_p[0][attrs[key]]
                P_1 *= dico_p[1][attrs[key]]
            else:
                #si la valeur de l'attribut n'existe pas dans l'enseble d'apprentissage,
                #alors sa probabilité conditionnelle vaut zero et on peut
                #faire une sortie anticipée
                return {0: 0.0, 1: 0.0}
        return {0: P_0, 1: P_1}
    
    def draw(self):
        """
        Construit un graphe orienté représentant naïve Bayes réduit.
        """
        tab_col = list(self.dico_P2D_l)
        res = ""
        for enfant in tab_col:
            res = res + "target" + "->" + enfant + ";"
        return utils.drawGraph(res[:-1])   

class ReducedMAPNaiveBayesClassifier(APrioriClassifier):
    """
    Classifieur par le maximum a posteriori en utilisant le modèle naïve Bayes réduit. 
    """
    def __init__(self, df, x):
        """
        Initialise le classifieur. Crée un dictionnarie où la clé est le nom de
        chaque attribut et la valeur est un dictionnaire de dictionnaires contenant
        les probabilités conditionnelles P(attribut | target) où attribut et target
        ne sont pas indépendants. Cree aussi un dictionnaire avec les probabilités
        de target = 0 et target = 1.
        
        :param df: dataframe. Doit contenir une colonne appelée "target" ne contenant que 0 ou 1.
        :param x: seuil de confiance pour le test de indépendance.
        """
        self.pTarget = {1: df["target"].mean()}
        self.pTarget[0] = 1 - self.pTarget[1] 
        self.dico_P2D_l = {}
        tab_col = list(df.columns.values)
        tab_col.remove("target")
        for attr in tab_col:
            if not isIndepFromTarget(df,attr,x):
                self.dico_P2D_l[attr] = P2D_l(df, attr)
    
    def estimClass(self, attrs):
        """
        À partir d'un dictionanire d'attributs, estime la classe 0 ou 1.        
        L'estimée est faite par maximum à posteriori à partir de dico_res.
        
        :param attrs: le dictionnaire nom-valeur des attributs
        :return: la classe 0 ou 1 estimée
        """
        dico_res = self.estimProbas(attrs)
        if dico_res[0] >= dico_res[1]:
            return 0
        return 1
        

    def estimProbas(self, attrs):
        """
        Calcule la probabilité à posteriori par naïve Bayes réduit: 
        P(target | attr1, ..., attrk).
        
        :param attrs: le dictionnaire nom-valeur des attributs
        """
        P_0 = self.pTarget[0]
        P_1 = self.pTarget[1]
        for key in self.dico_P2D_l:
            dico_p = self.dico_P2D_l[key]
            if attrs[key] in dico_p[0]:
                P_0 *= dico_p[0][attrs[key]]
                P_1 *= dico_p[1][attrs[key]]
            else:
                #si la valeur de l'attribut n'existe pas dans l'enseble d'apprentissage,
                #alors sa probabilité conditionnelle n'est pas definie et on peut
                #faire une sortie anticipée
                return {0: 0.0, 1: 0.0}
        P_0res = P_0 / (P_0 + P_1)
        P_1res = P_1 / (P_0 + P_1)
        return {0: P_0res, 1: P_1res}
    
    def draw(self):
        """
        Construit un graphe orienté représentant naïve Bayes réduit.
        """
        tab_col = list(self.dico_P2D_l)
        res = ""
        for enfant in tab_col:
            res = res + "target" + "->" + enfant + ";"
        return utils.drawGraph(res[:-1])   

#==============================================================================
# Question 7
#==============================================================================
def mapClassifiers(dic, df):
    """
    Représente graphiquement les classifiers à partir d'un dictionnaire dic de 
    {nom:instance de classifier} et d'un dataframe df, dans l'espace (précision,rappel). 
    
    :param dic: dictionnaire {nom:instance de classifier}
    :param df: dataframe. Doit contenir une colonne appelée "target" ne contenant que 0 ou 1.
    """
    precision = np.empty(len(dic))
    rappel = np.empty(len(dic))
    
    for i, nom in enumerate(dic):
         dico_stats = dic[nom].statsOnDF(df)
         precision[i] = dico_stats["Précision"]
         rappel[i] = dico_stats["Rappel"]
    
    fig, ax = plt.subplots()
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.set_xlabel("Précision")
    ax.set_ylabel("Rappel")
    ax.scatter(precision, rappel, marker = 'x', c = 'red') 
    
    for i, nom in enumerate(dic):
        ax.annotate(nom, (precision[i], rappel[i]))
    
    plt.show()
    
#==============================================================================
# Question 8
#==============================================================================
def MutualInformation(df, x, y):
    """
    """
    list_x = np.unique(df[x].values) # Valeurs possibles de x.
    list_y = np.unique(df[y].values) # Valeurs possibles de y.
    
    dico_x = {list_x[i]: i for i in range(list_x.size)} 
    #un dictionnaire associant chaque valeur a leur indice en list_x.
    
    dico_y = {list_y[i]: i for i in range(list_y.size)} 
    #un dictionnaire associant chaque valeur a leur indice en list_y.
    
    mat_xy = np.zeros((list_x.size, list_y.size), dtype = int)
    #matrice des valeurs P(x,y)
    
    for _, row in df.iterrows():
        i = row[x]
        j = row[y]
        mat_xy[dico_x[i], dico_y[j]]+= 1 
    
    mat_xy = mat_xy / mat_xy.sum()
    
    mat_x = mat_xy.sum(1)
    #matrice des P(x)
    mat_y = mat_xy.sum(0)
    #matrice des P(y)
    mat_px_py = np.dot(mat_x.reshape((mat_x.size, 1)),mat_y.reshape((1, mat_y.size))) 
    #matrice des P(x)P(y)
    
    mat_res = mat_xy / mat_px_py
    mat_res[mat_res == 0] = 1
    #pour éviter des problèmes avec le log de zero
    mat_res = np.log2(mat_res)
    mat_res *= mat_xy
    
    return mat_res.sum()
    
    
    
    
    
    