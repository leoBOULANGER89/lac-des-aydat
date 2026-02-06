import numpy as np
from scipy.spatial import cKDTree
from collections import defaultdict

# =====================================================
# Union-Find
# =====================================================
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0]*n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            self.parent[rx] = ry
        elif self.rank[rx] > self.rank[ry]:
            self.parent[ry] = rx
        else:
            self.parent[ry] = rx
            self.rank[rx] += 1
        return True

# =====================================================
# Fonctions OBJ
# =====================================================
def lire_obj(file_path):
    vertices, faces = [], []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.split()
            if not parts: continue
            if parts[0] == 'v':
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == 'f':
                faces.append([int(parts[1])-1, int(parts[2])-1, int(parts[3])-1])
    return vertices, faces

def sauvegarder_obj(file_path, vertices, faces):
    with open(file_path, 'w') as f:
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for tri in faces:
            f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")

# =====================================================
# Fusion rapide avec cKDTree
# =====================================================
def fusion_points_rapide(vertices, seuil_distance):
    vertices = np.array(vertices)
    tree = cKDTree(vertices)
    uf = UnionFind(len(vertices))

    # Trouver toutes les paires proches
    pairs = tree.query_pairs(r=seuil_distance)
    for i,j in pairs:
        uf.union(i,j)

    # Remplacer chaque groupe par le barycentre
    groups = defaultdict(list)
    for idx in range(len(vertices)):
        groups[uf.find(idx)].append(idx)

    for g in groups.values():
        if len(g) > 1:
            bary = np.mean(vertices[g], axis=0)
            for idx in g:
                vertices[idx] = bary

    # Liste des vertices à supprimer (tous sauf le premier de chaque groupe)
    L_retire = []
    for g in groups.values():
        if len(g) > 1:
            L_retire.extend(g[1:])
    return vertices.tolist(), L_retire, uf

# =====================================================
# Mise à jour des vertices et dictionnaire
# =====================================================
def update_vertices(L_retire, vertices):
    vertices_new = vertices[:]
    for i in sorted(L_retire, reverse=True):
        del vertices_new[i]
    dico = {}
    j = 0
    for i in range(len(vertices)):
        if i not in L_retire:
            dico[i] = j
            j += 1
    return vertices_new, dico

# =====================================================
# Mise à jour des faces
# =====================================================
def update_faces(faces, dico, uf):
    new_faces = []
    for f in faces:
        f_new = []
        for i in range(3):
            root = uf.find(f[i])
            if root not in dico:
                break  # sommet supprimé
            f_new.append(dico[root])
        if len(f_new) == 3 and len(set(f_new)) == 3:  # éviter triangles dégénérés
            new_faces.append(f_new)
    return new_faces

# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    fichier_entree = 'Lake_Aydat_Delaunay.obj'
    fichier_sortie = 'triangles_corriges.obj'
    seuil_distance = 40.0  # fusion des points proches

    vertices, faces = lire_obj(fichier_entree)
    print("Initial :", len(vertices), "vertices,", len(faces), "faces")

    vertices_fusion, L_retire, uf = fusion_points_rapide(vertices, seuil_distance)
    print("Après fusion :", len(vertices_fusion), "vertices,", len(faces), "faces")

    new_vertices, dico_indices = update_vertices(L_retire, vertices_fusion)
    new_faces = update_faces(faces, dico_indices, uf)

    sauvegarder_obj(fichier_sortie, new_vertices, new_faces)
    print("Mesh corrigé et indices mis à jour ✔")
