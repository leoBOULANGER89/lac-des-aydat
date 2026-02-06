import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict



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

def heron(a, b, c):
    p = (a+b+c)/2
    S = np.sqrt( p*(p-a)*(p-b)*(p-c) )
    return S

def select_faces(faces, vertices, seuil_aire):

    faces_a_decouper = []

    for i in range(len(faces)):

        a = distance(vertices[faces[i][0]], vertices[faces[i][1]])
        b = distance(vertices[faces[i][1]], vertices[faces[i][2]])
        c = distance(vertices[faces[i][2]], vertices[faces[i][0]])

        S = heron(a, b, c)

        if S > seuil_aire:
            l = [i, a, b, c]
            faces_a_decouper.append(l)

    return faces_a_decouper

def creer_faces(faces_a_decouper, faces, vertices):

    paires = []
    
    id_new_vertice = len(vertices)

    for f in faces_a_decouper:
        i, a, b, c = f[0], f[1], f[2], f[3]
        if a > b:
            if a > c:
                # a est le grand côté
                p_op = 2
                p_adj1 = 0
                p_adj2 = 1

            else:
                # c est le plus grand côté
                p_op = 1
                p_adj1 = 0
                p_adj2 = 2

        else:
            if c < b:
                # b est le plus grand côté
                p_op = 0
                p_adj1 = 1
                p_adj2 = 2

            else:
                # c est le plus grand côté
                p_op = 1
                p_adj1 = 0
                p_adj2 = 2


        new_vertice = [ (vertices[faces[i][p_adj1]][x] + vertices[faces[i][p_adj2]][x]) /2 for x in range(3) ]
        vertices.append(new_vertice)
        faces.append([faces[i][p_adj2], faces[i][p_op], id_new_vertice])
        paires.append((faces[i][p_adj1], faces[i][p_adj2], id_new_vertice))
        faces[i][p_adj2] = id_new_vertice
        id_new_vertice +=1

    return vertices, faces, paires


def retire_rez_de_chaussee(faces, vertices, coord, altitude):
    taille = len(faces)
    faces_rdc = []
    for i in range(taille):
        if vertices[faces[i][0]][coord] > seuil_altitude and vertices[faces[i][1]][coord] > seuil_altitude and vertices[faces[i][2]][coord] > seuil_altitude:
            faces_rdc.append(i)

    for i in sorted(faces_rdc, reverse=True):
        del faces[i]

    return faces


def corrige_faces(faces, vertices, paires):
    taille_paires = len(paires)
    taille_faces = len(faces)

    for i in range(taille_paires):
        print(i)
        facetrouvee = 0
        id_face = 0
        while facetrouvee == 0 and id_face < taille_faces:
            if paires[i][0] in faces[id_face] and paires[i][1] in faces[id_face]:
                facetrouvee = 1
                print("OUI !!!!")
                if faces[id_face][0] in paires[i]:
                    if faces[id_face][1] in paires[i]:
                        # points 1 et 2 en commun
                        pt_commun1 = 0
                        pt_commun2 = 1
                        pt_pas_commun = 2
                        
                    else:
                        # points 1 et 3 en commun
                        pt_commun1 = 0
                        pt_commun2 = 2
                        pt_pas_commun = 1
                
                else:
                    # points 2 et 3 en commun
                    pt_commun1 = 1
                    pt_commun2 = 2
                    pt_pas_commun = 0
                
                # Ajout de la face [2, new, autre]
                faces.append([faces[id_face][pt_commun2], paires[i][2], faces[id_face][pt_pas_commun]])

                # Changement du point pour avoir la face [1, new, autre]
                faces[id_face][pt_commun2] = paires[i][2]

                print(faces[id_face], '\t', faces[-1])

            id_face += 1
    return faces


    






vertices, faces = lire_obj('triangles4.obj')

coord = 1 # l'altitude est en y
seuil_altitude = 1.835

new_faces = retire_rez_de_chaussee(faces, vertices, coord, seuil_altitude)

seuil_aire = 1
faceadecouper = select_faces(faces, vertices, seuil_aire)

#print(faceadecouper)

new_vertices, new_faces, paires = creer_faces(faceadecouper, faces, vertices)
print(paires)

sauvegarder_obj('_avant.obj', new_vertices, new_faces)

new_faces2 = corrige_faces(new_faces, new_vertices, paires)

sauvegarder_obj('_apres.obj', new_vertices, new_faces2)