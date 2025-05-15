import re
import networkx as nx
import numpy as np


def read_Sizegraph(fileName):
    """Read the number of graphs in a file.
    Input: fileName (string) : the name of the file
    Ouptut: TAILLE (int) : the number of graphs in the file"""

    file = open(fileName, "r")
    nbGraph = 0
    for line in file:
        if line[0] == "t":
            nbGraph = nbGraph + 1
    return nbGraph


def load_graphs(fileName, TAILLE):
    """Load graphs from a file.
    args: fileName (string) : the name of the file)
    TAILLE (int) : the number of graphs in the file

    return: graphs (list of networkx graphs) : the list of graphs
    numbers (list of list of int) : the list of occurences of each graph
    nom (list of string) : the list of names of each graph)"""

    nbV = []
    nbE = []
    numbers = []
    noms = []
    namesPersons = []
    temptre = ""
    namesVertices = []
    for i in range(TAILLE):
        numbers.append([])
    ## Variables de stockage
    vertices = np.zeros(TAILLE)
    labelVertices = []
    edges = np.zeros(TAILLE)
    labelEdges = []
    compteur = -1
    numero = 0
    file = open(fileName, "r")
    for line in file:
        a = line
        b = a.split(" ")
        if b[0] == "t":
            compteur = compteur + 1
            if compteur > 0:
                noms.append(temptre)
                nbV.append(len(labelVertices[compteur - 1]))
                nbE.append(len(labelEdges[compteur - 1]))
                namesPersons.append(namesVertices)
            labelVertices.append([])
            labelEdges.append([])
            namesVertices = []
            val = b[2]
            val = re.sub("\n", "", val)
            val = int(val)
            numero = val
            temptre = ""
        if b[0] == "v":
            vertices[compteur] += 1

            # Récupération du label
            val = b[2]
            val = re.sub("\n", "", val)
            val = int(val)
            labelVertices[compteur].append(val)

            # Récupération du nom (après le # si présent)
            if "#" in line:
                name = line.split("#", 1)[-1].strip()
            else:
                name = f"v{vertices[compteur]-1}"  # nom générique si absent
            namesVertices.append(name)
        if b[0] == "e":
            edges[compteur] = edges[compteur] + 1
            num1 = int(b[1])
            num2 = int(b[2])
            val = b[3]
            val = re.sub("\n", "", val)
            val = int(val)
            labelEdges[compteur].append((num1, num2, val))
            temptre = temptre + line
        if b[0] == "x":
            temp = []
            for j in range(1, len(b)):
                if not (b[j] == "#"):
                    val = b[j]
                    val = re.sub("\n", "", val)
                    val = int(val)
                    temp.append(val)
            numbers[numero] = temp
    noms.append(temptre)
    nbV.append(len(labelVertices[compteur - 1]))
    nbE.append(len(labelEdges[compteur - 1]))
    namesPersons.append(namesVertices)
    graphes = []
    for i in range(len(vertices)):
        dicoNodes = {}
        graphes.append(nx.Graph())
        for j in range(int(vertices[i])):
            # tempDictionnaireNodes = {"color":labelVertices[i][j]}
            dicoNodes[j] = labelVertices[i][j]
        for j in range(int(edges[i])):
            graphes[i].add_edge(
                labelEdges[i][j][0], labelEdges[i][j][1], color=labelEdges[i][j][2]
            )
        graphes[i].add_nodes_from(
            [(node, {"color": attr}) for (node, attr) in dicoNodes.items()]
        )

    return graphes, numbers, namesPersons


def readLabels(fileLabel):
    """this function reads the file containing the labels of the graphs
        and convert them into 2 classes : 0 and 1

    Input : fileLabel (string) : the name of the file containing the labels
    Output : labels (list of int) : the list of labels of the graphs"""

    file = open(fileLabel, "r")
    labels = []
    numero = 0
    for line in file:
        lab = str(line).split()[0]
        if int(lab) == -1:
            labels.append(0)
        elif int(lab) > -1:
            labels.append(min(int(lab), 1))
        numero = numero + 1
    return labels
