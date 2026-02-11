import os
#from unionfind import unionfind
from unionfind import UnionFind
import csv
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


# ==========================================================
# =================== LECTURE / ECRITURE ===================
# ==========================================================

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
                elif parts[0] == 'f':
                    # Lignes avec les faces (avec les indices des points)
                    faces.append([int(parts[i].split("/")[0]) - 1 for i in range(1,4)])
    
    return vertices, faces

def sauvegarder_obj(file_path, vertices, faces):
    with open(file_path, 'w') as file:
        for v in vertices:
            file.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for f in faces:
            file.write(f"f {f[0]+1} {f[1]+1} {f[2]+1}\n")  # Remise des indices à 1 pour l'écriture


# ==========================================================
# ======================= OUTILS GEO =======================
# ==========================================================

def distance(v1, v2):
    """Calcule la distance Euclidienne entre deux points."""
    return np.sqrt((v1[0] - v2[0])**2 + (v1[1] - v2[1])**2 + (v1[2] - v2[2])**2)

def heron(a, b, c):
    p = (a+b+c)/2
    S = np.sqrt( p*(p-a)*(p-b)*(p-c) )
    return S


# ==========================================================
# =============== DIVISE GRANDS TRIANGLES ==================
# ==========================================================

def select_faces(faces, vertices, seuil_aire):
    faces_a_decouper = set()

    for i in range(len(faces)):

        a = distance(vertices[faces[i][0]], vertices[faces[i][1]])
        b = distance(vertices[faces[i][1]], vertices[faces[i][2]])
        c = distance(vertices[faces[i][2]], vertices[faces[i][0]])

        S = heron(a, b, c)

        if S > seuil_aire:
            l = (i, a, b, c)
            faces_a_decouper.add(l)

    return faces_a_decouper

def creer_faces(faces_a_decouper, faces, vertices):

    paires = []
    
    id_new_vertice = len(vertices)

    for i, a, b, c in faces_a_decouper:
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


def corrige_faces(faces, vertices, paires):
    taille_paires = len(paires)
    taille_faces = len(faces)

    for i in range(taille_paires):
        facetrouvee = 0
        id_face = 0
        while facetrouvee == 0 and id_face < taille_faces:
            if paires[i][0] in faces[id_face] and paires[i][1] in faces[id_face]:
                facetrouvee = 1
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


            id_face += 1
    return faces


def suppr_grand (IN_PATH_OBJ, OUT_PATH_OBJ, seuil_aire = 1):
    vertices, faces = lire_obj(IN_PATH_OBJ)

    faceadecouper = select_faces(faces, vertices, seuil_aire)

    new_vertices, new_faces, paires = creer_faces(faceadecouper, faces, vertices)

    new_faces2 = corrige_faces(new_faces, new_vertices, paires)

    sauvegarder_obj(OUT_PATH_OBJ, new_vertices, new_faces2)



# ==========================================================
# ========== SUPPRESSION DES PETITS TRIANGLES ==============
# ==========================================================

def fusionner_points(vertices, seuil_distance):
    """
    Fusionne les sommets proches en les plaçant à leur centre.
    Construit en parallèle une structure Union-Find reliant les sommets fusionnés.

    Retourne :
    - vertices modifiés (toutes)
    - vertices qui on été modifiés
    - UnionFind
    """

    n = len(vertices)
    uf = UnionFind(range(n))
    vertices_mod = set()

    modification = True

    while modification:
        modification = False

        for i in range(n):
            for j in range(i + 1, n):
                ri = uf.find(i)
                rj = uf.find(j)

                if ri == rj:
                    continue

                p1 = vertices[ri]
                p2 = vertices[rj]

                if distance(p1, p2) < seuil_distance:
                    milieu = [
                        (p1[0] + p2[0]) / 2,
                        (p1[1] + p2[1]) / 2,
                        (p1[2] + p2[2]) / 2,
                    ]

                    vertices[ri] = milieu
                    vertices[rj] = milieu
                    uf.union(ri, rj)

                    vertices_mod.add(i)
                    vertices_mod.add(j)
                    modification = True

    return vertices, vertices_mod, uf


def delete_vertices(vertices, vertices_mod, uf):
    n = len(vertices)
    L_retire = []

    # Déterminer quels indices retirer ou garder
    to_remove = set()
    for i in vertices_mod:
        ri = uf.find(i)
        if ri != i:
            to_remove.add(i)
            L_retire.append(i)

    L_garde = set()
    # Reconstruire vertices en place, une seule passe
    j = 0  # pointeur pour les éléments à garder
    for i in range(n):
        if i not in to_remove:
            vertices[j] = vertices[i]
            L_garde.add(i)

            j += 1

    del vertices[j:]

    return vertices, L_retire, L_garde


def dichotomie(L_retire, v):
    a = 0
    L_retire.sort()
    if v < L_retire[-1]:
        if L_retire[0] < v:
            b = len(L_retire)
            while b-a > 1 :
                m = (a + b) // 2
                if L_retire[m] < v:
                    a = m
                else:
                    b = m
            a += 1
    else:
        a = len(L_retire)
    return a 


def creer_dico(L_retire, L_garde):
    dico = {}
    for i in L_garde:
        dico[i]=i-dichotomie(L_retire, i)
    return dico
    
def modif_forme(faces, dico, uf):
    n = len(faces)
    to_remove = set()

    for i in range(len(faces)):
        for j in range(3):
            faces[i][j] = dico[uf.find(faces[i][j])]

        if len(set(faces[i])) < 3:
            to_remove.add(i)
    
    j = 0  # pointeur pour les éléments à garder
    for i in range(n):
        if i not in to_remove:
            faces[j] = faces[i]
            j += 1

    del faces[j:]


    return faces


# ==========================================================
# ======================== PIPELINE ========================
# ==========================================================

def pipeline(IN_PATH_OBJ,
             OUT_PATH_OBJ,
             seuil_aire=10,
             seuil_distance=5):
    """
    Pipeline complet :

    1) Découpe des grands triangles
    2) Fusion des petits triangles
    3) Reconstruction des faces
    4) Sauvegarde
    """

    print("Lecture du maillage...")
    vertices, faces = lire_obj(IN_PATH_OBJ)

    # ---------------------------
    # 1️⃣ Suppression grands triangles
    # ---------------------------
    print("Découpe des grands triangles...")
    faces_a_decouper = select_faces(faces, vertices, seuil_aire)

    vertices, faces, paires = creer_faces(faces_a_decouper, faces, vertices)
    faces = corrige_faces(faces, vertices, paires)

    print("Nombre de sommets après découpe :", len(vertices))
    print("Nombre de faces après découpe :", len(faces))

    # ---------------------------
    # 2️⃣ Suppression petits triangles
    # ---------------------------
    print("Fusion des petits triangles...")
    vertices, vertices_mod, uf = fusionner_points(vertices, seuil_distance)

    vertices, L_retire, L_garde = delete_vertices(vertices, vertices_mod, uf)

    dico = creer_dico(L_retire, L_garde)

    faces = modif_forme(faces, dico, uf)

    print("Nombre final de sommets :", len(vertices))
    print("Nombre final de faces :", len(faces))

    # ---------------------------
    # 3️⃣ Sauvegarde
    # ---------------------------
    print("Sauvegarde du maillage...")
    sauvegarder_obj(OUT_PATH_OBJ, vertices, faces)

    print("Pipeline terminé avec succès.")






# ==========================================================
# ========================== EXEC ==========================
# ==========================================================
if __name__ == "__main__":

    pipeline(
        "../resultat/Lake_Aydat/Lake_Aydat_Delaunay.obj",
        "../resultat/Lake_Aydat/mesh_final.obj",
    )





#'../resultat/Lake_Aydat/Lake_Aydat_Delaunay.obj'
#'../resultat/Lake_Aydat/triangles4.obj'