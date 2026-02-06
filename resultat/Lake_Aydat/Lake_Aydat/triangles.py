import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

class UnionFind:
    def __init__(self, n):
        # parent[i] = parent de i (au début, i est son propre parent)
        self.parent = list(range(n))
        # rank = approximation de la profondeur de l’arbre
        self.rank = [0] * n

    def find(self, x):
        # Compression de chemin
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return False  # déjà dans le même ensemble

        # Union par rang
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

        return True

    def connected(self, x, y):
        return self.find(x) == self.find(y)





def lire_obj(file_path):
    vertices = []
    faces = []
    
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split(" ")
            if parts:
                if parts[0] == 'v':  # Lignes avec les coordonnées des vertices
                    # Convertir les coordonnées en flottants
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif parts[0] == 'f':  # Lignes avec les faces
                    faces.append([int(parts[1])-1, int(parts[2])-1, int(parts[3])-1])
    
    return vertices, faces

def sauvegarder_obj(file_path, vertices, faces):
    with open(file_path, 'w') as file:
        for v in vertices:
            file.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for f in faces:
            file.write(f"f {f[0]+1} {f[1]+1} {f[2]+1}\n")  # Remise des indices à 1 pour l'écriture

def distance(v1, v2):
    """Calcule la distance Euclidienne entre deux points."""
    return np.sqrt((v1[0] - v2[0])**2 + (v1[1] - v2[1])**2 + (v1[2] - v2[2])**2)



def fusionner_points2(vertices, seuil_distance):
    """Fusionne les points proches en les déplaçant au centre de chaque paire proche, sans changer les faces."""
    
    # On parcourt chaque paire de points dans les vertices
    arretprec = 0
    arret = 1
    compteur = 0
    verticesdelete = []
    while arret > 0 and arretprec != arret:
        arretprec = arret
        arret = 0
        for i, p1 in enumerate(vertices):
            for j in range(i + 1, len(vertices)):
                p2 = vertices[j]
                # Si la distance entre deux points est inférieure au seuil, on les fusionne
                if distance(p1, p2) < seuil_distance:
                    # Déplacer les deux points à leur position médiane
                    vertices[i] = [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2, (p1[2] + p2[2]) / 2]
                    vertices[j] = vertices[i]  # Les deux points sont déplacés à la même position
                    #verticesdelete.append(i)
                    if (i,j) not in verticesdelete:
                        verticesdelete.append((i,j))
                    arret += 1
        compteur += 1
    verticesdelete.sort()
    return vertices, verticesdelete


def deletevertices(edges, nbnoeuds):

    
    uf = UnionFind(nbnoeuds+1)

    for a, b in edges:
        uf.union(a, b)

    groups = defaultdict(list)

    for i in range(nbnoeuds+1):
        groups[uf.find(i)].append(i)

    """for group in groups.values():
        if len(group) > 1:
            print(group)"""

    return groups, uf


def f_garde(group):
    l_garde = []
    for i in group.keys():
        l_garde.append(i)
    return l_garde


def f_retire(group):
    l_retire = []
    for i in group.values():
        if len(i) > 1:
            l_retire.extend(i[1:])
    return l_retire


def dichotomie(Lretire, v):
    a = 0
    L_retire.sort()
    if v < L_retire[-1]:
        if L_retire[0] < v:
            b = len(Lretire)
            while b-a > 1 :
                m = (a + b) // 2
                if Lretire[m] < v:
                    a = m
                else:
                    b = m
            if (Lretire[a]>=v):
                print(Lretire[a], v)
            a += 1
    else:
        a = len(L_retire)
    return a 


def creerdico(L_retire, L_garde):
    dico = {}
    for i in L_garde:
        dico[i]=i-dichotomie(L_retire, i)
    return dico
    
def modifforme(faces, dico, uf):
    for i in range(len(faces)):
        #print(faces[i], end = "\t")
        for j in range(3):
            faces[i][j] = dico[uf.find(faces[i][j])]
            #faces[i][j] = uf.find(faces[i][j])
        #print(faces[i])
                
    return faces

def modifpoint(L_retire, vertices):
    for i in sorted(L_retire, reverse=True):
        del vertices[i]
    return vertices





# Charger le fichier OBJ
vertices, faces = lire_obj('Lake_Aydat_Delaunay.obj')


# Appliquer la fusion des points proches avec un seuil de distance
seuil_distance = 40  # Choisir un seuil approprié
vertices_fusionnes, verticesdelete = fusionner_points2(vertices, seuil_distance)

sauvegarder_obj('triangles2.obj', vertices_fusionnes, faces)

#print(verticesdelete)
#print(deletevertices(verticesdelete))
nbnoeuds = len(vertices)

group, uf = deletevertices(verticesdelete, nbnoeuds)

#print(f_garde(group))
#print(f_retire(group))
L_retire = f_retire(group)
print(L_retire)
L_garde = f_garde(group)
#L_garde_new = remetvertices(f_retire(group), f_garde(group))
#print(L_garde_new)
dico = creerdico(L_retire, L_garde)
print(L_garde)
#print(dico)

new_vertices = modifpoint(L_retire, vertices)
sauvegarder_obj('triangles3.obj', new_vertices, faces)

new_faces = modifforme(faces, dico, uf)
# Sauvegarder le fichier modifié
#sauvegarder_obj('triangles4.obj', new_vertices, faces)

sauvegarder_obj('triangles4.obj', new_vertices, new_faces)
