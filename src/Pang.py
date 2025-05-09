from networkx.algorithms import isomorphism
from networkx.algorithms.isomorphism import ISMAGS
import copy
import networkx as nx
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pandas as pd
import tqdm
from sklearn.model_selection import StratifiedKFold
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.svm import SVC
import rbo
import scipy as sp
from sklearn.cluster import AgglomerativeClustering
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score


import sys, getopt
from sklearn import metrics

def read_Sizegraph(fileName):
    """Read the number of graphs in a file.
    Input: fileName (string) : the name of the file
    Ouptut: TAILLE (int) : the number of graphs in the file"""
    
    file = open(fileName, "r")
    nbGraph=0
    for line in file:
       if line[0]=="t":
            nbGraph=nbGraph+1
    return nbGraph

def load_graphs(fileName,TAILLE):
    """Load graphs from a file.
    args: fileName (string) : the name of the file)
    TAILLE (int) : the number of graphs in the file
    
    return: graphs (list of networkx graphs) : the list of graphs
    numbers (list of list of int) : the list of occurences of each graph
    nom (list of string) : the list of names of each graph)"""
    
    nbV=[]
    nbE=[]
    numbers = []
    noms = []
    for i in range(TAILLE):
        numbers.append([])
    ## Variables de stockage
    vertices = np.zeros(TAILLE)
    labelVertices = []
    edges = np.zeros(TAILLE)
    labelEdges = []
    compteur=-1
    numero=0
    file = open(fileName, "r")
    for line in file:
        a = line
        b = a.split(" ")
        if b[0]=="t":
            compteur=compteur+1
            if compteur>0:
                noms.append(temptre)
                nbV.append(len(labelVertices[compteur-1]))
                nbE.append(len(labelEdges[compteur-1]))
            labelVertices.append([])
            labelEdges.append([])
            val = b[2]
            val = re.sub("\n","",val)
            val = int(val)
            numero = val
            temptre=""
        if b[0]=="v":
            vertices[compteur]=vertices[compteur]+1
            val = b[2]
            val = re.sub("\n","",val)
            val = int(val)
            labelVertices[compteur].append(val)
            temptre=temptre+line
        if b[0]=="e":
            edges[compteur]=edges[compteur]+1
            num1 = int(b[1])
            num2 = int(b[2])
            val = b[3]
            val = re.sub("\n","",val)
            val = int(val)
            labelEdges[compteur].append((num1,num2,val))
            temptre=temptre+line
        if b[0]=="x":
            temp= []
            for j in range(1,len(b)):
                if not(b[j]=="#"):
                    val = b[j]
                    val = re.sub("\n","",val)
                    val = int(val)
                    temp.append(val)
            numbers[numero]=temp  
    noms.append(temptre)
    nbV.append(len(labelVertices[compteur-1]))
    nbE.append(len(labelEdges[compteur-1]))
    graphes = []
    for i in range(len(vertices)):
        dicoNodes = {}
        graphes.append(nx.Graph())
        for j in range(int(vertices[i])):
            #tempDictionnaireNodes = {"color":labelVertices[i][j]}
            dicoNodes[j]=labelVertices[i][j]
        for j in range(int(edges[i])):
            graphes[i].add_edge(labelEdges[i][j][0],labelEdges[i][j][1],color=labelEdges[i][j][2])
        graphes[i].add_nodes_from([(node, {'color': attr}) for (node, attr) in dicoNodes.items()])
    return graphes,numbers,noms

def load_patterns(fileName,TAILLE):
    """ This function loads the post-processed patterns, i.e with occurences.
    fileName (string) : the name of the file
    TAILLE (int) : the number of graphs in the file
    
    return: graphs (list of networkx graphs) : the list of patterns
            numbers (list of list of int) : the list of occurences of each graph
            numberoccurences (list of list of int) : the list of occurences of each pattern
    """
    numbers = []
    numberoccurences = []
    numbercoverage = []
    noms = []
    for i in range(TAILLE):
        numbers.append([])
        numberoccurences.append([])
        numbercoverage.append([])
    ## Variables de stockage
    vertices = np.zeros(TAILLE)
    labelVertices = []
    edges = np.zeros(TAILLE)
    labelEdges = []
    compteur=-1
    numero=0
    file = open(fileName, "r")
    for line in file:
        a = line
        b = a.split(" ")
        if b[0]=="t":
            compteur=compteur+1
            if compteur>0:
                noms.append(temptre)
            labelVertices.append([])
            labelEdges.append([])
            val = b[2]
            val = re.sub("\n","",val)
            val = int(val)
            numero = val
            temptre=""
        if b[0]=="v":
            vertices[compteur]=vertices[compteur]+1
            val = b[2]
            val = re.sub("\n","",val)
            val = int(val)
            labelVertices[compteur].append(val)
            temptre=temptre+line
        if b[0]=="e":
            edges[compteur]=edges[compteur]+1
            num1 = int(b[1])
            num2 = int(b[2])
            val = b[3]
            val = re.sub("\n","",val)
            val = int(val)
            labelEdges[compteur].append((num1,num2,val))
            temptre=temptre+line
        if b[0]=="x":
            temp= []
            tempOccu = []
            tempCoverage = []
            for j in range(1,len(b)-1):
                val = b[j]
                val = re.sub("\n","",val)
                if not(val=="#" or val==""):
                    val = str(val).split("/")
                    numeroGraph = int(val[0])
                    val = str(val[1]).split(":")
                    coverage=1
                    if len(val)>1:
                        coverage = float(val[1])
                    occurences = int(float(val[0]))
                    temp.append(numeroGraph)
                    tempOccu.append(occurences)
                    tempCoverage.append(coverage)
            numbers[numero]=temp 
            numberoccurences[numero]=tempOccu
            numbercoverage[numero]=tempCoverage
    noms.append(temptre)
    graphes = []
    for i in range(len(vertices)):
        dicoNodes = {}
        graphes.append(nx.Graph())
        for j in range(int(vertices[i])):
            dicoNodes[j]=labelVertices[i][j]
        for j in range(int(edges[i])):
            graphes[i].add_edge(labelEdges[i][j][0],labelEdges[i][j][1],color=labelEdges[i][j][2])
        graphes[i].add_nodes_from([(node, {'color': attr}) for (node, attr) in dicoNodes.items()])
    return graphes,numbers,numberoccurences


def load_patterns_processed(fileName, TAILLE):
     # Initialize data structures
    numbers = [{} for _ in range(TAILLE)]  # List of occurrences with monomorphisms and isomorphisms
    noms = []  # List of names
    labelVertices = [[] for _ in range(TAILLE)]  # List of vertices for each pattern
    labelEdges = [[] for _ in range(TAILLE)]  # List of edges for each pattern
    compteur = -1  # Counter for current pattern
    with open(fileName, "r") as file:
        for line in file:
            b = line.split(" ") 
            if b[0] == "t":
                # New pattern starts, store previous data if available
                if compteur >= 0:
                    noms.append(temptre)
                # Reset variables for the new pattern
                compteur += 1
                temptre = ""
                labelVertices[compteur] = []
                labelEdges[compteur] = []
                
                # Extract motif id
                val = int(b[2].strip())
                numero = val
            
            elif b[0] == "v":
                # Vertex information
                val = int(b[2].strip())  # The label of the vertex
                labelVertices[compteur].append(val)
                temptre += line  # Append the line to the pattern's description
                
            elif b[0] == "e":
                # Edge information
                num1 = int(b[1])  # First vertex
                num2 = int(b[2])  # Second vertex
                val = int(b[3].strip())  # The edge label
                labelEdges[compteur].append((num1, num2, val))
                temptre += line  # Append the line to the pattern's description
                
            elif b[0] == "x":
                # Pattern occurrences with monomorphisms and isomorphisms
                temp = {}
                for val in b[1:len(b)-1]:
                    # Format: graph_id/monomorphisms:isomorphisms
                    graph_weight_pair = val.split("/")
                    graph_id = int(graph_weight_pair[0].strip())
                    monomorph_isomorph = graph_weight_pair[1].split(":")
                    monomorphisms = float(monomorph_isomorph[0].strip())  # Number of monomorphisms
                    isomorphisms = float(monomorph_isomorph[1].strip())  # Number of isomorphisms
                    
                    temp[graph_id] = (monomorphisms, isomorphisms)  # Store both counts as a tuple
                numbers[numero] = temp
    
    # Store the last pattern data
    noms.append(temptre)
    numbers[compteur] = temp  # Store the last pattern's occurrences
    
    return numbers



    
def patternMeasures(keep,labels,id_graphs,TAILLEPATTERN):
    lenC = 0
    lennotC = 0
    for i in range(len(keep)):
        if labels[keep[i]]==1:
            lenC = lenC+1
        else:
            lennotC = lennotC+1
    lenALL = lenC+lennotC

    pC = lenC/lenALL
    pnotC = 1 - pC
    pP = np.zeros(TAILLEPATTERN)
    pnotP = np.zeros(TAILLEPATTERN)
    pPC = np.zeros(TAILLEPATTERN)
    pPnotC = np.zeros(TAILLEPATTERN)
    pnotPC = np.zeros(TAILLEPATTERN)
    pnotPnotC = np.zeros(TAILLEPATTERN)
    pPassumingC = np.zeros(TAILLEPATTERN)
    pPassumingnotC = np.zeros(TAILLEPATTERN)
    pnotPassumingC = np.zeros(TAILLEPATTERN)
    pnotPassumingnotC = np.zeros(TAILLEPATTERN)
    pCassumingP = np.zeros(TAILLEPATTERN)
    pCassumingnotP = np.zeros(TAILLEPATTERN)
    pnotCassumingP = np.zeros(TAILLEPATTERN)
    pnotPnotC = np.zeros(TAILLEPATTERN)
    pnotCassumingnotP = np.zeros(TAILLEPATTERN)
    toConsider = np.zeros(TAILLEPATTERN)
    t11 = np.zeros(TAILLEPATTERN)
    t12 = np.zeros(TAILLEPATTERN)
    t21 = np.zeros(TAILLEPATTERN)
    t22 = np.zeros(TAILLEPATTERN)
    for i in range(TAILLEPATTERN):
        t_Pos = 0
        t_Neg = 0
        for j in range(len(id_graphs[i])):
            if j in keep:
                if labels[id_graphs[i][j]] == 1:
                    t_Pos += 1
                else:
                    t_Neg += 1
        toConsider[i]=1
        if t_Pos+t_Neg==0:
            toConsider[i]=0
        t11[i] = t_Pos
        t12[i] = t_Neg
        t21[i] = lenC-t_Pos
        t22[i] = lennotC-t_Neg


        pP[i] = (t_Pos+t_Neg)/lenALL

        pnotP[i] = 1-pP[i]


        if lenALL == 0:
            pPC[i] = 0
            pPnotC[i] = 0
        else:
            pPC[i] = t_Pos/lenALL
            pPnotC[i] = (t_Neg)/lenALL

        pnotPC[i] = (lenC-t_Pos)/lenALL
        pnotPnotC[i] = (lennotC-t_Neg)/lenALL
        if lenC != 0:
            pPassumingC[i]= t_Pos/lenC      
        else:
            pPassumingC[i]= 0
        if lennotC != 0:
            pPassumingnotC[i]= t_Neg/lennotC
        else:
            pPassumingnotC[i]= 0
        if lenC != 0:
            pnotPassumingC[i]= (lenC-t_Pos)/lenC
        else:
            pnotPassumingC[i]= 0
        if lennotC != 0:
            pnotPassumingnotC[i]= (lennotC-t_Neg)/lennotC
        else:
            pnotPassumingnotC[i]= 0

        if t_Pos+t_Neg==0:
            pCassumingP[i]= 0
            pnotCassumingP[i]= 0
        else:
            pCassumingP[i]= t_Pos/(t_Pos+t_Neg)
            pnotCassumingP[i]= t_Neg/(t_Pos+t_Neg)
        
        if t_Pos+t_Neg==lenALL:
            pCassumingnotP[i]= 0
            pnotCassumingnotP[i]= 0
        else:
            pCassumingnotP[i]= (lenC-t_Pos)/(lenALL-t_Pos-t_Neg)
            pnotCassumingnotP[i]= (lennotC-t_Neg)/(lenALL-t_Pos-t_Neg)
    
    ds = DiscriminationScores(toConsider,lenALL,pP,pnotP,pC,pnotC,pPC,pPnotC,pnotPC,pnotPnotC,pPassumingC,pPassumingnotC,pnotPassumingC,pnotPassumingnotC,pCassumingP,pCassumingnotP,pnotCassumingP,pnotCassumingnotP,t11,t12,t21,t22)
    
    return ds


#### Function for discrimination scores

class DiscriminationScores:
    def __init__(self,toConsider,lenALL,pP,pnotP,pC,pnotC,pPC,pPnotC,pnotPC,pnotPnotC,pPassumingC,pPassumingnotC,pnotPassumingC,pnotPassumingnotC,pCassumingP,pCassumingnotP,pnotCassumingP,pnotCassumingnotP,t11,t12,t21,t22):
        self.toConsider = toConsider
        self.lenALL = lenALL
        self.pP = pP
        self.pnotP = pnotP
        self.pC = pC
        self.pnotC = pnotC
        self.pPC = pPC
        self.pPnotC = pPnotC
        self.pnotPC = pnotPC
        self.pnotPnotC = pnotPnotC
        self.pPassumingC = pPassumingC
        self.pPassumingnotC = pPassumingnotC
        self.pnotPassumingC = pnotPassumingC
        self.pnotPassumingnotC = pnotPassumingnotC
        self.pCassumingP = pCassumingP
        self.pCassumingnotP = pCassumingnotP
        self.pnotCassumingP = pnotCassumingP
        self.pnotCassumingnotP = pnotCassumingnotP
        self.t11 = t11
        self.t12 = t12
        self.t21 = t21
        self.t22 = t22
    
def Acc(discriminationScore):
    return discriminationScore.pPC + discriminationScore.pnotPnotC

def Brins(discriminationScore):
    numerator = discriminationScore.pP * discriminationScore.pnotC
    denominator = discriminationScore.pPnotC 

    # Use np.where to handle the two special cases
    result = np.where(denominator == 0, float('inf'), numerator / denominator)
    result = np.where(numerator == 0, 0, result)

    return result
def CConf(discriminationScore):
    return discriminationScore.pCassumingP - discriminationScore.pC

def Cole(discriminationScore):
    return (discriminationScore.pCassumingP - discriminationScore.pC)/(1-discriminationScore.pC)

def ColStr(discriminationScore):
    term1num= (discriminationScore.pPC +discriminationScore.pnotPnotC)
    term1denom = (discriminationScore.pP*discriminationScore.pC+discriminationScore.pnotP*discriminationScore.pnotC)
    term2num = 1 - discriminationScore.pP*discriminationScore.pC - discriminationScore.pnotC*discriminationScore.pnotP
    term2denom = 1 - discriminationScore.pPC - discriminationScore.pnotCassumingnotP
    # Use np.where to handle the two special cases for term1 and term2
    term1 = np.where(term1denom == 0, float('inf'), term1num / term1denom)
    term1 = np.where(term1num == 0, 0, term1)
    term2 = np.where(term2denom == 0, float('inf'), term2num / term2denom)
    term2 = np.where(term2num == 0, 0, term2)
    return term1 * term2

def Conf(discriminationScore):
    return discriminationScore.pCassumingP

def Cos(discriminationScore):
    product = discriminationScore.pCassumingP * discriminationScore.pPassumingC
    # Use np.where to handle the special case
    result = np.where(product == 0, 0, np.sqrt(product))
    return result

def Cover(discriminationScore):
    return discriminationScore.pPassumingC

def Dep(discriminationScore):
    return np.abs(discriminationScore.pnotCassumingP - discriminationScore.pnotC)

def Excex(discriminationScore):
    numerator = discriminationScore.pnotCassumingP
    denominator = discriminationScore.pCassumingnotP

    # Use np.where to handle the special case
    result = np.where(denominator == 0, -1*float('inf'), 1 - (numerator / (denominator)))
    result = np.where(numerator == 0, 1, result)
    return result

def Gain(discriminationScore):
    term1 = discriminationScore.pCassumingP
    term2 = np.log(discriminationScore.pC)

    result = np.where(term1 == 0, -1*float('inf'), discriminationScore.pPC * (np.log(term1) - term2))
    result = np.where(discriminationScore.pPC == 0, 0, result)

    return result
    
def GR(discriminationScore):
    numerator = discriminationScore.pPassumingC
    denominator = discriminationScore.pPassumingnotC

    # Use np.where to handle the special case
    result = np.where(denominator == 0, float('inf'), numerator / denominator)
    result = np.where(numerator == 0, 0, result)

    return result

def gTest(discriminationScore):

    term1 = discriminationScore.pPassumingC * np.log(discriminationScore.pPassumingC / discriminationScore.pPassumingnotC)
    term2 = (1-discriminationScore.pPassumingC) * np.log((1-discriminationScore.pPassumingC) / (1-discriminationScore.pPassumingnotC))

    return term1 + term2

def InfGain(discriminationScore):
    term1 = -np.log(discriminationScore.pC)
    term2 = discriminationScore.pCassumingP

    # Use np.where to handle the special case
    result = np.where(term2 == 0, -1*float('inf'), term1 * np.log(term2))
    return result

def Jacc(discriminationScore):
    numerator = discriminationScore.pPC
    denominator = discriminationScore.pP + discriminationScore.pC - discriminationScore.pPC

    # Use np.where to handle the special case
    result = np.where(denominator == 0, float('inf'), numerator / denominator)
    result = np.where(numerator == 0, 0, result)

    return result

def Klos(discriminationScore):
    term1 = np.sqrt(discriminationScore.pPC)
    term2 = discriminationScore.pCassumingP - discriminationScore.pC

    # Use np.where to handle the special case
    result = np.where(term1 == 0, 0, term1 * term2)

    return result

def Lap(discriminationScore):
    return (discriminationScore.pPC + 1/discriminationScore.lenALL)/(discriminationScore.pP + 2/discriminationScore.lenALL)

def Lever(discriminationScore):
    return discriminationScore.pPC-(discriminationScore.pC*discriminationScore.pP)


def Lift(discriminationScore):
    numerator = discriminationScore.pPC
    denominator = discriminationScore.pP * discriminationScore.pC

    # Use np.where to handle the special case
    result = np.where(denominator == 0, float('inf'), numerator / denominator)
    result = np.where(numerator == 0, 0, result)
    return result

def MDisc(discriminationScore):
    numerator = discriminationScore.pPC * discriminationScore.pnotPnotC 
    denominator = discriminationScore.pPnotC * discriminationScore.pnotPC 

    result = np.where(denominator == 0, float('inf'), numerator / denominator)
    result = np.where(numerator == 0, -1*float('inf'), np.log(result))

    return result

def MutInf(discriminationScore):
    alpha = discriminationScore.pPC *np.log(discriminationScore.pPC/(discriminationScore.pP*discriminationScore.pC+0.0000001)+0.0000001)
    beta = discriminationScore.pnotPC *np.log(discriminationScore.pnotPC/(discriminationScore.pnotP*discriminationScore.pC+0.0000001)+0.0000001)
    gamma = discriminationScore.pPnotC *np.log(discriminationScore.pPnotC/(discriminationScore.pP*discriminationScore.pnotC+0.0000001)+0.0000001)
    delta = discriminationScore.pnotPnotC *np.log(discriminationScore.pnotPnotC/(discriminationScore.pnotP*discriminationScore.pnotC+0.0000001)+0.0000001)
    return alpha+beta+gamma+delta

def NetConf(discriminationScore):
    return (discriminationScore.pCassumingP-discriminationScore.pC)/(1-discriminationScore.pP+0.0000001)

def OddsR(discriminationScore):
    alpha = np.where(discriminationScore.pPC == 1 , float('inf'),discriminationScore.pPC/(1-discriminationScore.pPC))
    alpha = np.where(discriminationScore.pPC == 0 , 0 , alpha)
    beta = np.where(discriminationScore.pPnotC == 1 , float('inf'),discriminationScore.pPnotC/(1-discriminationScore.pPnotC))
    beta = np.where(discriminationScore.pPnotC == 0 , 0 , beta)
    return alpha/beta

def Pearson(discriminationScore):
    return (discriminationScore.pPC-discriminationScore.pP*discriminationScore.pC)/np.sqrt(discriminationScore.lenALL*discriminationScore.pP*discriminationScore.pC*discriminationScore.pnotP*discriminationScore.pnotC)

def RelRisk(discriminationScore): 
    numerator = discriminationScore.pCassumingP 
    denominator = discriminationScore.pCassumingnotP 

    # Use np.where to handle the special case
    result = np.where(denominator == 0, float('inf'), numerator / denominator)
    result = np.where(numerator == 0, 0, result)

    return result

def Sebag(discriminationScore): 
    numerator = discriminationScore.pPC
    denominator = discriminationScore.pPnotC

    # Use np.where to handle the special case
    result = np.where(denominator == 0, float('inf'), numerator / denominator)
    result = np.where(numerator == 0, 0, result)

    return result

def Spec(discriminationScore): 
    return discriminationScore.pnotCassumingnotP

def Strenght(discriminationScore): 
    numerator = GR(discriminationScore)
    denominator = numerator + 1 

    # Use np.where to handle the special case
    result = np.where(numerator == np.inf, discriminationScore.pPC , (numerator / denominator)* discriminationScore.pPC)
    return result

def Supp(discriminationScore): 
    return discriminationScore.pPC

def SuppDif(discriminationScore): 
    return discriminationScore.pPassumingC - discriminationScore.pPassumingnotC

def SuppDifAbs(discriminationScore): 
    return np.abs(discriminationScore.pPassumingC - discriminationScore.pPassumingnotC)

def WRACC(discriminationScore): 
    return discriminationScore.pP * (discriminationScore.pCassumingP-discriminationScore.pC)

def chiTwo(discriminationScore): 
    return discriminationScore.lenALL*(discriminationScore.pPC*discriminationScore.pnotPnotC-discriminationScore.pPnotC*discriminationScore.pnotPC)**2/(discriminationScore.pP*discriminationScore.pC*discriminationScore.pnotP*discriminationScore.pnotC)

def Zhang(discriminationScore): 
    maxi = np.ones(len(discriminationScore.pP))
    for i in range(len(discriminationScore.pP)):
        maxi[i] = max(discriminationScore.pPC[i]*discriminationScore.pnotC,discriminationScore.pC*discriminationScore.pPnotC[i])
    return (discriminationScore.pPC - discriminationScore.pP*discriminationScore.pC)/(maxi)


def TPR(discriminationScore):
    #return discriminationScore.pCassumingP
    return discriminationScore.pCassumingP

def FPR(discriminationScore):
    result = np.where(discriminationScore.pCassumingnotP == 0, float('inf'), 1/discriminationScore.pCassumingnotP)
    return result


def CertaintyFactor(discriminationScore):
    return (discriminationScore.pCassumingP - discriminationScore.pC) / (1 - discriminationScore.pC)


#####
# Les mesures à rajouter : 
def Gini(discriminationScore):
    gini_index = 1 - (discriminationScore.pCassumingP ** 2 + discriminationScore.pnotCassumingP ** 2)
    return 1/(gini_index+0.0000000001)

def Gini2(discriminationScore):
    gini_index = (discriminationScore.pPassumingC ** 2) * discriminationScore.pCassumingP + (discriminationScore.pPassumingnotC ** 2) * discriminationScore.pnotCassumingP
    return 1/(gini_index+0.0000000001)

def Entropy(discriminationScore):
    epsilon = 1e-10  # Avoid log(0)
    p0 = discriminationScore.pnotCassumingP
    p1 = discriminationScore.pCassumingP
    
    entropy = - (p0 * np.log2(p0 + epsilon) + p1 * np.log2(p1 + epsilon))
    return 1 / (entropy + epsilon)

def Fisher(discriminationScore):
    epsilon = 1e-10  # Avoid division by zero
    mean_diff = (discriminationScore.pCassumingP - discriminationScore.pnotCassumingP) ** 2
    var_sum = discriminationScore.pCassumingP * (1 - discriminationScore.pCassumingP) + \
              discriminationScore.pnotCassumingP * (1 - discriminationScore.pnotCassumingP)
    
    return mean_diff / (var_sum + epsilon)


#Ficher Score
#https://arxiv.org/pdf/1202.3725
#https://www.researchgate.net/publication/376154712_Feature_selection_techniques_for_machine_learning_a_survey_of_more_than_two_decades_of_research
#https://dl.acm.org/doi/pdf/10.1145/3136625






def creationDictionnaryScores():
    dico = {"AbsSuppDif": SuppDifAbs}
    return {k: dico[k] for k in sorted(dico)}



def readLabels(fileLabel):
    """ this function reads the file containing the labels of the graphs
        and convert them into 2 classes : 0 and 1
        
    Input : fileLabel (string) : the name of the file containing the labels
    Output : labels (list of int) : the list of labels of the graphs"""
    
    file=open(fileLabel,"r")
    labels = []
    types = []
    numero=0
    for line in file:
        lab = str(line).split()[0]
        if int(lab)==-1:
            labels.append(0)
        elif int(lab)>-1:
            labels.append(min(int(lab),1))
        numero=numero+1
    return labels,labels

def graphKeep(Graphes,labels):
    """Equilibrate the number of graphs in each class"""
    ### Equilibre dataset
    if len(labels)-sum(labels)>sum(labels):
        minority=1
        NbMino=sum(labels)
    else:
        minority =0
        NbMino=len(labels)-sum(labels)
    keep = []
    NbMino = 0
    count=0
    graphs=[]
    for i in range(len(labels)):
        if labels[i]==minority:
            NbMino=NbMino+1
            keep.append(i)
    complete=NbMino
    for i in range(len(labels)):   
        if labels[i]!=minority:
            if count<complete:
                count=count+1
                keep.append(i)

    return keep



def metricDotProduct(X):
    a = (len(X[0])-np.matmul(X,np.transpose(X)))/2
    return a

def ComputeRepresentation(keep, keepPatterns, id_graphs, labels):
    nb_graphs = len(keep)
    nb_patterns = len(keepPatterns)
    
    rep_binary = np.zeros((nb_graphs, nb_patterns))

    
    graph_index = {g: idx for idx, g in enumerate(keep)}
    pattern_index = {p: idx for idx, p in enumerate(keepPatterns)}

    for p in keepPatterns:
        for j in id_graphs[p]:
            if j not in graph_index:
                continue
            i = graph_index[j]
            k = pattern_index[p]
            rep_binary[i, k] = 1

    newLabels = [labels[i] for i in keep]
    return rep_binary,newLabels

def performClustering(pattern,distance):
    model = AgglomerativeClustering(distance_threshold=distance,metric="precomputed",n_clusters=None,linkage="complete")
    model = model.fit(pattern)
    return model


def deduplicate_patterns(id_graphsMono):
    seen = set()
    patternsUnique = []
    dejaVu = []
    for i, graphs in enumerate(id_graphsMono):
        if not graphs:
            continue
        key = frozenset(graphs)
        if key not in seen:
            patternsUnique.append(i)
            dejaVu.append(graphs)
            seen.add(key)
    return patternsUnique, dejaVu



def build_super_matrix(dejaVu, TAILLEGRAPHE):
    superMatrice = np.full((len(dejaVu), TAILLEGRAPHE), -1, dtype=np.int8)
    for i, graph_ids in enumerate(dejaVu):
        superMatrice[i, graph_ids] = 1
    return superMatrice

#import cdist
from scipy.spatial.distance import cdist
def selectCurrentClustering(pattern, distance, id_graphs, convertisseur, nbPointPerCluster, TypeMedoid, superMatrice):
    """ Perform full clustering at a given distance threshold and select representative patterns."""
    
    newID_graphs = []
    convertisseur = {}
    res = []
    
    model = performClustering(pattern, distance)
    clusters = model.labels_
    n_clusters = max(clusters) + 1
    
    # Pour chaque cluster
    for cluster_id in range(n_clusters):
        # Récupérer les indices globaux des motifs du cluster
        id_clusters_points = [i for i, label in enumerate(clusters) if label == cluster_id]
        
        # Récupérer les superMatrice associés
        cluster_points = [superMatrice[i] for i in id_clusters_points]
        
        if not cluster_points:
            continue  # cluster vide (très rare mais au cas où)
        
        cluster_points = np.vstack(cluster_points)
        
        for k in range(min(nbPointPerCluster, len(cluster_points))):
            # Calcul du centroïde
            cluster_centroid = np.mean(cluster_points, axis=0)
            distances = cdist(cluster_points, [cluster_centroid])

            if TypeMedoid == "M":
                idx_selected = np.argmin(distances)
            else:  # TypeMedoid == "F"
                idx_selected = np.argmax(distances)
            
            # On utilise id_graphs pour retrouver l'index GLOBAL du motif
            res.append(id_clusters_points[idx_selected])
            newID_graphs.append(id_graphs[id_clusters_points[idx_selected]])

            # Supprimer ce point pour éviter de le reprendre
            cluster_points = np.delete(cluster_points, idx_selected, axis=0)
            id_clusters_points.pop(idx_selected)

    return model, res, convertisseur, newID_graphs


def load_dataset(arg,mode):
    if mode == "c":
        motifs = "CLOSED"
    else:
        motifs = "GENERAUX"
    folder="../data/"+str(arg)+"/"
    FILEGRAPHS=folder+str(arg)+"_graphs.txt"
    if motifs == "GENERAUX":
        FILESUBGRAPHS=folder+str(arg)+"_patterns.txt"
    if motifs == "CLOSED":
        FILESUBGRAPHS=folder+str(arg)+"_CGSPAN.txt"
    FILELABEL =folder+str(arg)+"_labels.txt"
    FILEMONOMORPHISM=folder+str(arg)+"_Occu.txt"
    FILEISO = folder+str(arg)+"_patternInduced.txt"
    
    #FILEISO = folder+str(arg)+"_patternInduced.txt"
    TAILLEGRAPHE=read_Sizegraph(FILEGRAPHS)
    TAILLEPATTERN=read_Sizegraph(FILESUBGRAPHS)

    
    """loading graphs"""
    print("Reading graphs")
    Graphes,useless_var,PatternsRed= load_graphs(FILEGRAPHS,TAILLEGRAPHE)
    """loading patterns"""
    print("Reading patterns")
    Subgraphs,id_graphs,noms = load_graphs(FILESUBGRAPHS,TAILLEPATTERN)


    labelss,types = readLabels(FILELABEL)

    keep = range(len(Graphes))

    return id_graphs,labelss,keep,TAILLEGRAPHE

def partialRepresentation(X,patterns):
    return X[:,np.array(patterns)]

def AlterateMetric(metric,patternsAGarder):
    for i in range(len(metric)):
        if i not in patternsAGarder:
            metric[i]=-1000000
    return metric


def prepareSuperMatrix(id_graphsMono, TAILLEGRAPHE):
    """Optimized preparation of patterns and super matrix."""
    # Convert lists of graphs into frozensets to ignore order
    frozen_id_graphs = [frozenset(g) for g in id_graphsMono]

    # Détection des motifs uniques
    seen = {}
    patternsUnique = []
    for idx, motif in enumerate(frozen_id_graphs):
        if motif not in seen:
            seen[motif] = len(patternsUnique)
            patternsUnique.append(idx)

    # Construction de la superMatrice
    superMatrice = np.full((len(patternsUnique), TAILLEGRAPHE), -1, dtype=np.int8)
    for mat_idx, pattern_idx in enumerate(patternsUnique):
        for node in id_graphsMono[pattern_idx]:
            superMatrice[mat_idx, node] = 1

    return patternsUnique, superMatrice


def load_labels_with_genres(path):
    labels_bin = {}
    genres = {}
    genre_set = set()
    with open(path, "r") as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) < 3:
                continue
            label = int(tokens[0])
            gid = int(tokens[1])
            genre = tokens[2].lower()
            labels_bin[gid] = label
            genres[gid] = genre
            genre_set.add(genre)
    return labels_bin, genres, sorted(genre_set)

def top_motifs_per_genre(keep, id_graphsMono, all_labels, all_genres, genre_list, nb_top=5):
    """
    Pour chaque genre : extrait les motifs les plus discriminants (selon un score choisi).
    
    Args:
        keep: structure interne pour patternMeasures
        id_graphsMono: liste des motifs par graphe
        all_labels: dict {gid: 0/1}
        all_genres: dict {gid: genre}
        genre_list: liste des genres uniques
        nb_top: nombre de motifs à retourner
    """
    from collections import defaultdict

    results = {}

    # Liste unique des motifs
    dejaVu = []
    for motifs in id_graphsMono:
        if motifs not in dejaVu:
            dejaVu.append(motifs)

    for genre in genre_list:
        # Construction d'un label binaire : 1 si genre courant, 0 sinon
        y_bin = []
        for gid in range(len(id_graphsMono)):
            g_genre = all_genres.get(gid, "unknown")
            y_bin.append(1 if g_genre == genre else 0)
        print(sum(y_bin))

        # Calcul des scores de discrimination
        discriminationScores = patternMeasures(keep, y_bin, dejaVu, len(dejaVu))
        dicoScores = creationDictionnaryScores()

        # Utilise la première mesure par défaut (par ex. Sup, ou autre)
        score_fn = list(dicoScores.values())[0]
        score_vals = score_fn(copy.deepcopy(discriminationScores))
        score_vals = AlterateMetric(score_vals, list(range(len(dejaVu))))
        score_vals = np.array(score_vals)

        # Nettoyage des -2 (non considérés)
        for i in range(len(score_vals)):
            if discriminationScores.toConsider[i] == -2:
                score_vals[i] = -1e6

        # Top-k motifs
        top_ids = np.argsort(score_vals)[::-1][:nb_top]
        results[genre] = [(i, round(score_vals[i], 3)) for i in top_ids]

    return results


id_graphs,labels,keep,TAILLEGRAPHE = load_dataset("DRACOR","g")

# Représenter les graphes en fonction des motifs dans les top motifs
rep_binary,newlabels = ComputeRepresentation(keep,range(0,len(id_graphs)),id_graphs, labels)

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

#Classer les motifs 
DiscriminationScores = patternMeasures(keep,labels,id_graphs,len(id_graphs))
dicoScores = creationDictionnaryScores()
score_fn = list(dicoScores.values())[0]
score_vals = score_fn(copy.deepcopy(DiscriminationScores))
score_vals = AlterateMetric(score_vals, list(range(len(id_graphs))))
score_vals = np.array(score_vals)
print(score_vals)

for i in range(1,101):
    #Garder les i motifs avec les meilleurs scores
    top_ids = np.argsort(score_vals)[::-1][:i]
    rep_binary = partialRepresentation(rep_binary, top_ids)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(rep_binary, newlabels)
    y_pred = model.predict(rep_binary)
    print(f"Pour i motifs : {i} : {accuracy_score(newlabels, y_pred)}")




# Appliquer la classification

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(rep_binary, newlabels)

# Évaluation
# Obtenir toutes les prédictions via CV
y_pred = cross_val_predict(model, rep_binary, newlabels, cv=5)

# Évaluation complète
print("✅ Accuracy :", accuracy_score(newlabels, y_pred))
print("🎯 F1-score macro :", f1_score(newlabels, y_pred, average='macro'))
print("🔍 Rapport complet :")
print(classification_report(newlabels, y_pred, target_names=["Classe 0", "Classe 1"]))

#Matrice de confusion
y_pred = model.predict(rep_binary)
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(newlabels, y_pred)
print(conf_matrix)

import matplotlib.pyplot as plt
from collections import defaultdict

def plot_genre_accuracy_histogram(y_true, y_pred, graph_ids, gid_to_genre):
    """
    Affiche un histogramme du taux de bonne classification par genre de film.
    
    Args:
        y_true (list of int): Labels vrais (0 ou 1).
        y_pred (list of int): Labels prédits (0 ou 1).
        graph_ids (list of int): ID des graphes testés.
        gid_to_genre (dict): {graph_id: genre_text}.
    """
    genre_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    for gid, yt, yp in zip(graph_ids, y_true, y_pred):
        genre = gid_to_genre.get(gid, "unknown")
        genre_stats[genre]["total"] += 1
        if yt == yp:
            genre_stats[genre]["correct"] += 1

    genres = sorted(genre_stats.keys())
    totals = [genre_stats[g]["total"] for g in genres]
    accuracies = [genre_stats[g]["correct"] / genre_stats[g]["total"] if genre_stats[g]["total"] > 0 else 0 for g in genres]

    # Affichage
    plt.figure(figsize=(10, 5))
    bars = plt.bar(genres, accuracies)
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Taux de bonne classification")
    plt.title("📊 Précision par genre de film")
    
    # Ajouter le texte au-dessus des barres
    for bar, acc, total in zip(bars, accuracies, totals):
        plt.text(bar.get_x() + bar.get_width() / 2, acc + 0.02, f"{acc:.2f}\n({total})", 
                 ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.show()

# Afficher la matrice de confusion
