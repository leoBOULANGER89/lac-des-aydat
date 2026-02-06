import numpy as np
from collections import defaultdict, deque

# =====================================================
# OBJ I/O
# =====================================================
def lire_obj(path):
    vertices, faces = [], []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("v "):
                _, x, y, z = line.split()
                vertices.append([float(x), float(y), float(z)])
            elif line.startswith("f "):
                _, a, b, c = line.split()
                faces.append([int(a)-1, int(b)-1, int(c)-1])
    return vertices, faces


def sauvegarder_obj(path, vertices, faces):
    with open(path, "w") as f:
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for a, b, c in faces:
            f.write(f"f {a+1} {b+1} {c+1}\n")

# =====================================================
# GEOMETRIE
# =====================================================
def longueur(v1, v2):
    return np.linalg.norm(np.array(v1) - np.array(v2))

def aire_triangle(A, B, C):
    return 0.5 * np.linalg.norm(
        np.cross(np.array(B)-np.array(A), np.array(C)-np.array(A))
    )

def milieu(A, B):
    return [(A[i] + B[i]) / 2 for i in range(3)]

# =====================================================
# SUBDIVISION PAR LONGUEUR (ANTI-TROUS)
# =====================================================
def subdiviser_par_longueur(vertices, faces, longueur_max):
    verts = vertices[:]
    current_faces = faces[:]

    while True:
        edge_mid = {}
        new_faces = []
        subdivised = False

        def get_mid(i, j):
            key = tuple(sorted((i, j)))
            if key in edge_mid:
                return edge_mid[key]
            m = milieu(verts[i], verts[j])
            idx = len(verts)
            verts.append(m)
            edge_mid[key] = idx
            return idx

        for a, b, c in current_faces:
            A, B, C = verts[a], verts[b], verts[c]

            if max(longueur(A,B), longueur(B,C), longueur(C,A)) <= longueur_max:
                new_faces.append([a, b, c])
                continue

            subdivised = True
            ab = get_mid(a, b)
            bc = get_mid(b, c)
            ca = get_mid(c, a)

            new_faces.extend([
                [a, ab, ca],
                [ab, b, bc],
                [ca, bc, c],
                [ab, bc, ca]
            ])

        current_faces = new_faces
        if not subdivised:
            break

    return verts, current_faces

# =====================================================
# SUPPRESSION TRIANGLES TROP PETITS
# =====================================================
def filtrer_petits_triangles(vertices, faces, aire_min):
    return [
        f for f in faces
        if aire_triangle(vertices[f[0]], vertices[f[1]], vertices[f[2]]) >= aire_min
    ]

# =====================================================
# DETECTION DES TROUS
# =====================================================
def trouver_aretes_frontiere(faces):
    edges = defaultdict(int)
    for a, b, c in faces:
        for e in [(a,b), (b,c), (c,a)]:
            edges[tuple(sorted(e))] += 1
    return [e for e, n in edges.items() if n == 1]

def trouver_boucles(aretes):
    adj = defaultdict(list)
    for a, b in aretes:
        adj[a].append(b)
        adj[b].append(a)

    boucles = []
    visited = set()

    for start in adj:
        if start in visited:
            continue
        loop = []
        current = start
        prev = None
        while True:
            loop.append(current)
            visited.add(current)
            voisins = [v for v in adj[current] if v != prev]
            if not voisins:
                break
            nxt = voisins[0]
            prev, current = current, nxt
            if current == start:
                break
        if len(loop) >= 3:
            boucles.append(loop)
    return boucles

# =====================================================
# REBOUCHAGE DES TROUS
# =====================================================
def reboucher_trous(vertices, faces):
    aretes = trouver_aretes_frontiere(faces)
    boucles = trouver_boucles(aretes)

    new_faces = faces[:]

    for boucle in boucles:
        centre = np.mean([vertices[i] for i in boucle], axis=0).tolist()
        idx_centre = len(vertices)
        vertices.append(centre)

        for i in range(len(boucle)):
            a = boucle[i]
            b = boucle[(i+1) % len(boucle)]
            new_faces.append([a, b, idx_centre])

    return vertices, new_faces

# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    fichier_entree = "Lake_Aydat_Delaunay.obj"
    fichier_sortie = "triangles_clean.obj"

    longueur_max = 1000.0
    aire_min = 50.0

    vertices, faces = lire_obj(fichier_entree)
    print("Initial :", len(vertices), "v,", len(faces), "f")

    vertices, faces = subdiviser_par_longueur(vertices, faces, longueur_max)
    print("Après subdivision :", len(vertices), "v,", len(faces), "f")

    faces = filtrer_petits_triangles(vertices, faces, aire_min)
    print("Après suppression petits triangles :", len(faces), "faces")

    vertices, faces = reboucher_trous(vertices, faces)
    print("Après rebouchage :", len(vertices), "v,", len(faces), "f")

    sauvegarder_obj(fichier_sortie, vertices, faces)
    print("Mesh corrigé, étanche, réparé ✔")
