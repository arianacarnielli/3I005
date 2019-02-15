# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 17:00:17 2019

@author: arian
"""

import tictactoe_etu as tictac

j1 = tictac.AgentAlea()
j2 =tictac.AgentAlea()

state = tictac.MorpionState()

jeu = tictac.Jeu(state, j1, j2)

jeu.run(draw=True)