import numpy as np
import random as rd
import copy
import matplotlib.pyplot as plt
import matplotlib.patches as patches

## Constante
OFFSET = 0.2


class State:
    """ Etat generique d'un jeu de plateau. Le plateau est represente par une matrice de taille NX,NY,
    le joueur courant par 1 ou -1. Une case a 0 correspond a une case libre.
    * next(self,coup) : fait jouer le joueur courant le coup.
    * get_actions(self) : renvoie les coups possibles
    * win(self) : rend 1 si le joueur 1 a gagne, -1 si le joueur 2 a gagne, 0 sinon
    * stop(self) : rend vrai si le jeu est fini.
    * fonction de hashage : renvoie un couple (matrice applatie des cases, joueur courant).
    """
    NX,NY = None,None
    def __init__(self,grid=None,courant=None):
        self.grid = copy.deepcopy(grid) if grid is not None else np.zeros((self.NX,self.NY),dtype="int")
        self.courant = courant or 1
    def next(self,coup):
        pass
    def get_actions(self):
        pass
    def win(self):
        pass
    def stop(self):
        pass
    @classmethod
    def fromHash(cls,hash):
        return cls(np.array([int(i)-1 for i in list(hash[0])],dtype="int").reshape((cls.NX,cls.NY)),hash[1])
    def hash(self):
        return ("".join(str(x+1) for x in self.grid.flat),self.courant)
            
class Jeu:
    """ Jeu generique, qui prend un etat initial et deux joueurs.
        run(self,draw,pause): permet de joueur une partie, avec ou sans affichage, avec une pause entre chaque coup. 
                Rend le joueur qui a gagne et log de la partie a la fin.
        replay(self,log): permet de rejouer un log
    """
    def __init__(self,init_state = None,j1=None,j2=None):
        self.joueurs = {1:j1,-1:j2}
        self.state = copy.deepcopy(init_state)
        self.log = None
    def run(self,draw=False,pause=0.5):
        log = []
        if draw:
            self.init_graph()
        while not self.state.stop():
            coup = self.joueurs[self.state.courant].get_action(self.state)
            log.append((self.state,coup))
            self.state = self.state.next(coup)
            if draw:
                self.draw(self.state.courant*-1,coup)
                plt.pause(pause)
        return self.state.win(),log
    def init_graph(self):
        self._dx,self._dy  = 1./self.state.NX,1./self.state.NY
        self.fig, self.ax = plt.subplots()
        for i in range(self.state.grid.shape[0]):
            for j in range(self.state.grid.shape[1]):
                self.ax.add_patch(patches.Rectangle((i*self._dx,j*self._dy),self._dx,self._dy,\
                        linewidth=1,fill=False,color="black"))
        plt.show(block=False)
    def draw(self,joueur,coup):
        color = "red" if joueur>0 else "blue"
        self.ax.add_patch(patches.Rectangle(((coup[0]+OFFSET)*self._dx,(coup[1]+OFFSET)*self._dy),\
                        self._dx*(1-2*OFFSET),self._dy*(1-2*OFFSET),linewidth=1,fill=True,color=color))
        plt.draw()
    def replay(self,log,pause=0.5):
        self.init_graph()
        for state,coup in log:
            self.draw(state.courant,coup)
            plt.pause(pause)

class MorpionState(State):
    """ Implementation d'un etat du jeu du Morpion. Grille de 3X3. 
    """
    NX,NY = 3,3
    def __init__(self,grid=None,courant=None):
        super(MorpionState,self).__init__(grid,courant)
    def next(self,coup):
        state =  MorpionState(self.grid,self.courant)
        state.grid[coup]=self.courant
        state.courant *=-1
        return state
    def get_actions(self):
        return list(zip(*np.where(self.grid==0)))
    def win(self):
        for i in [-1,1]:
            if ((i*self.grid.sum(0))).max()==3 or ((i*self.grid.sum(1))).max()==3 or ((i*self.grid)).trace().max()==3 or ((i*np.fliplr(self.grid))).trace().max()==3: return i
        return 0
    def stop(self):
        return self.win()!=0 or (self.grid==0).sum()==0
    def __repr__(self):
        return str(self.hash())


class Puissance4State(State):
    """ Implementation d'un etat du jeu Puissance 4. Grille de 7X6. 
    """
    NX,NY = 7,6
    def __init__(self,grid=None,courant=None):
        super(Puissance4State,self).__init__(grid,courant)
    def next(self,coup):
        state =  Puissance4State(self.grid,self.courant)
        state.grid[coup]=self.courant
        state.courant *=-1
        return state
    
    def get_actions(self):   
        pos = []   
        for x in range(self.NX):
            for y in range(self.NY):
                if self.grid[x][y] == 0:
                    pos.append((x, y))
                    break
        return pos
    
    def win(self):
        #for i in [-1,1]:
        for x in range (self.NX):
            for y in range (self.NY - 4):
                if (self.grid[x][y] == self.grid[x][y + 1] == self.grid[x][y + 2] == self.grid[x][y + 3]):
                    return self.grid[x][y]
        for y in range (self.NY):
            for x in range (self.NX - 4):
                if (self.grid[x][y] == self.grid[x + 1][y] == self.grid[x + 2][y] == self.grid[x + 3][y]):
                    return self.grid[x][y]  
        
        for x in range (self.NX - 4):
            for y in range (self.NY - 4):
                if (self.grid[x][y] == self.grid[x + 1][y + 1] == self.grid[x + 2][y + 2] == self.grid[x + 3][y + 3]):
                    return self.grid[x][y]  
               
        for y in range (self.NY - 4):
            for x in range (self.NX - 4):
                if (self.grid[x][y] == self.grid[x + 1][y + 1] == self.grid[x + 2][y + 2] == self.grid[x + 3][y + 3]):
                    return self.grid[x][y]  
        return 0
    
    def stop(self):
        return self.win()!=0 or (self.grid==0).sum()==0
    def __repr__(self):
        return str(self.hash())

class Agent:
    """ Classe d'agent generique. Necessite une methode get_action qui renvoie l'action correspondant
    a l'etat du jeu state"""
    def __init__(self):
        pass
    def get_action(self,state):
        pass

class AgentAlea(Agent):
    """
    """
    def __init__(self):
        super(AgentAlea, self).__init__()
        
    def get_action(self, state):
        coups_possibles = state.get_actions()
        return rd.choice(coups_possibles)
    
    
class AgentMC(Agent):
    """
    """
    def __init__(self, n = 5):
        super(AgentMC, self).__init__()
        self.N = n
    
    def get_action(self, state):
        coups_possibles = state.get_actions()
        vict = dict.fromkeys(coups_possibles, 0)
        total = dict.fromkeys(coups_possibles, 0)
        
        j1 = AgentAlea()
        #On garantit que chaque coup est joué au moins une fois.
        for cp in coups_possibles:
            state_try = state.next(cp)
            jeu = Jeu(state_try, j1, j1)
            victoire, _ = jeu.run()
            vict[cp] += victoire * state.courant
            total[cp] += 1

        for i in range(len(coups_possibles) * self.N):
            cp = rd.choice(coups_possibles)
            state_try = state.next(cp)
            jeu = Jeu(state_try, j1, j1)
            victoire, _ = jeu.run()
            vict[cp] += victoire * state.courant
            total[cp] += 1
        
        coups = sorted(vict, key=(lambda key:vict[key]/total[key]), reverse=True)
        return coups[0]

class AgentMTTS(Agent):
    """
    """
    def __init__(self, n = 20):
        super(AgentMTTS, self).__init__()
        self.N = n
    
    def get_action(self, state):
        #initialisation du noeud racine
        racine = Noeud(state)
        #initialisation du joueur aleatoire
        j1 = AgentAlea()
        
        #recuperation des coups possibles a partir de state
        coups_possibles = state.get_actions()
        
        #creation de noeuds representant tous les coups possibles en sortant de la racine
        #on joue une fois aléatoirement a partir des coups possibles pour remplir les noeuds
        for i in range(len(coups_possibles)):
            state_try = state.next(coups_possibles[i])
            enfant = Noeud(state_try, racine)
            racine.kids[coups_possibles[i]] = enfant
            jeu = Jeu(state_try, j1, j1)
            victoire, _ = jeu.run()
            enfant.maj(-victoire)
        
        for i in range(self.N * len(coups_possibles)):
            nd = racine
            #print("wins racine =" + str(nd.wins))
            #print("total racine =" + str(nd.total))
            nd = nd.choix_ucb()
            jeu = Jeu(nd.state, j1, j1)
            victoire, _ = jeu.run()
            #print(nd.state.courant)
            nd.maj(-victoire)
       # print("fin d'un tour \n")
        return max(racine.kids, key = lambda k: racine.kids[k].loss/racine.kids[k].total)
#==============================================================================
#        for key,val in racine.kids.items():
#            print(key)
#            print(val.wins) 
#           print(val.wins/val.total)
#        print("fin des tests \n")
#        print(max(racine.kids, key = lambda k: racine.kids[k].wins/racine.kids[k].total))
#       print("fin d'un tour \n")
#==============================================================================
#       return max(racine.kids, key = lambda k: racine.kids[k].wins/racine.kids[k].total)
        

class Noeud:
    """
    """
    def __init__(self, state, parent = None):
        self.state = state
        self.parent = parent
        self.kids = {}
        self.loss = 0
        self.total = 0
        
    def maj(self, i):
        self.loss += i * self.state.courant
        self.total += 1
        if self.parent is not None:
            self.parent.maj(i)
        
    def choix_ucb(self):
        #recuperation du state lie au noeud courant
        state_par = self.state
        #si on est dans un state terminal, on retourne le noeud courant
        if state_par.stop():
            return self
        #on teste si la liste des enfants de ce noeud a ete deja initialise.
        #si c'est pas le cas, on l'initialise
        if self.kids == {}:
            cp = state_par.get_actions()
            for i in range (len(cp)):
                state_try = state_par.next(cp[i])
                enfant = Noeud(state_try, self)
                self.kids[cp[i]] = enfant 
        #s'il y a un enfant qui n'a pas encore ete testé (total = 0), on le retourne
        for i in self.kids:
            if self.kids[i].total == 0:
                return self.kids[i]
        #tous les enfants on ete visités au moins un fois, on aplique l'algo UCB
        #et on appele la fonction de façon recursive sur l'enfant choisi    
        t = sum([noeud.total for noeud in self.kids.values()])
        #print(max(self.kids, key = lambda k: self.kids[k].wins / self.kids[k].total + np.sqrt(2*np.log(t)/ self.kids[k].total)))
        return self.kids[max(self.kids, key = lambda k: self.kids[k].loss / self.kids[k].total + np.sqrt(2*np.log(t)/ self.kids[k].total))].choix_ucb()
        
#        keys = []
#        mu = []
#        Na = []
#             
#        for key,val in self.kids.items():
#            keys.append(key)
#            mu.append(val.wins/val.total)
#            Na.append(val.total)
#            
#        mu = np.array(mu)
#        Na = np.array(Na)
#        t = Na.sum()
#        index = np.argmax(mu + np.sqrt(2*np.log(t)/Na))
#        
#        return self.kids[keys[index]].choix_ucb()        
        

    
    
    