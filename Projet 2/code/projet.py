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
        for t in df.itertuples():
            dic=t._asdict()
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
        à partir d'un dictionanire d'attributs, estime la classe 0 ou 1
        
        L'estimée est faite par maximum de vraisemblance à partir de dico_p.
        
        :param attrs: le  dictionnaire nom-valeur des attributs
        :return: la classe 0 ou 1 estimée
        """
        val = attrs[self.attr] 
        if self.dico_p[0][val] >= self.dico_p[1][val]:
            return 0
        return 1

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
        à partir d'un dictionanire d'attributs, estime la classe 0 ou 1
        
        L'estimée est faite par maximum a posteriori à partir de dico_p.
        
        :param attrs: le  dictionnaire nom-valeur des attributs
        :return: la classe 0 ou 1 estimée
        """
        val = attrs[self.attr] 
        if self.dico_p[val][0] >= self.dico_p[val][1]:
            return 0
        return 1
    
    
def nbParams(df, liste = None):
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
    

def nbParamsIndep(df):
    taille = 0
    liste = list(df.columns) 
    
    for col in liste:
        taille += (np.unique(df[col].values).size * 8)
    
    res = str(len(liste)) + " variable(s) : " + str(taille) + " octets"
    if taille >= 1024:
        res = res + " = " + octetToStr(taille)
    print (res)
    
    