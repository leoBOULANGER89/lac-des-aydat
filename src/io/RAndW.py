"""
Module RAndW – Fonctions utilitaires de lecture et d'écriture.

Ce module regroupe un ensemble de fonctions utilisées dans différents scripts du projet (triangulation de Delaunay, analyse de maillages, reconstruction, etc.).

Il centralise les opérations de :
    - vérification d'existence de fichiers,
    - lecture de fichiers CSV (points, segments),
    - écriture de fichiers CSV,
    - lecture et extraction de données depuis des fichiers OBJ,
    - calcul de métriques géométriques élémentaires (longueurs d'arêtes, aires de triangles).

Objectif
--------

Éviter la duplication de code dans les scripts principaux en isolant toutes les opérations d'I/O (Input/Output) et
les traitements géométriques de base dans un module commun.

Organisation des fonctions
--------------------------

Lecture :
    - check_files_exist
    - read_points
    - read_curves
    - load_obj_for_dico

Écriture :
    - save_CSV

Formats pris en charge
----------------------

CSV :
    - Points 3D : colonnes "x", "y", "z"
    - Segments  : colonnes "id", "ix", "iy", "depth"

OBJ :
    - Sommets : lignes commençant par "v"
    - Faces triangulaires : lignes commençant par "f"

Notes
-----

- Les index du format OBJ (1-based) sont convertis en indexation Python (0-based).
- Les fonctions ne gèrent pas d'interface utilisateur ; elles sont destinées à être appelées par d'autres scripts.
- Les fichiers de sortie sont écrasés s'ils existent déjà.
"""



import os
import csv
import numpy as np
import pandas as pd
from collections import defaultdict

# =============================================================================================================
# CHECK
# =============================================================================================================
def check_files_exist(INPUT_FILES):
    """
    Vérifie l'existence d'une liste de fichiers.

    Parameters
    ----------
    INPUT_FILES : list of str
        Liste des chemins de fichiers à vérifier.

    Returns
    -------
    None
        La fonction ne retourne rien si tous les fichiers existent.

    Raises
    ------
    FileNotFoundError
        Si au moins un des fichiers spécifiés n'existe pas.

    Notes
    -----
    - La vérification est effectuée via os.path.isfile.
    - L'exception est levée dès le premier fichier manquant.
    """

    for file in INPUT_FILES:
        if not os.path.isfile(file):
            raise FileNotFoundError(f"Le fichier image n'existe pas : {file}")
        

# =============================================================================================================
# CSV
# =============================================================================================================

# ------------------------
# LECTURE CSV
# ------------------------
def read_curves(INPUT_CSV_POINTS, INPUT_CSV_LINES):
    """
    Lit un fichier de points 3D et un fichier de segments associés.

    Cette fonction :
        1. Charge les points 3D depuis un CSV,
        2. Regroupe les segments par identifiant de courbe,
        3. Associe une profondeur (depth) à chaque courbe.

    Parameters
    ----------
    INPUT_CSV_POINTS : str
        Chemin vers le fichier CSV contenant les points (x, y, z).
    INPUT_CSV_LINES : str
        Chemin vers le fichier CSV contenant les segments sous la forme : (id, ix, iy, depth).

    Returns
    -------
    points : np.ndarray
        Tableau de shape (N, 3) contenant les coordonnées des points.
    curves : list of list of tuple(int, int)
        Liste indexée par ID contenant les segments (i0, i1).
        L'indice -1 regroupe les segments ayant un ID négatif.
    depths : list of float or None
        Liste des profondeurs associées à chaque courbe.

    Raises
    ------
    ValueError
        - Si le fichier de points est vide.
        - Si le fichier de segments est vide.

    Notes
    -----
    - Le header des fichiers CSV est ignoré.
    - Les indices de segments correspondent aux indices des points dans le tableau retourné.
    - Si plusieurs profondeurs sont définies pour un même ID, la dernière valeur lue est conservée.
    """
    
    # --- Lecture des points ---
    pts = []
    with open(INPUT_CSV_POINTS, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            pts.append((float(row[0]), float(row[1]), float(row[2])))

    if not pts:
        raise ValueError("Le fichier de points est vide")

    # --- Regroupement des segments par ID ---
    curves_dict = defaultdict(list)
    depths_dict = {}

    with open(INPUT_CSV_LINES, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header

        for row in reader:
            curve_id = int(row[0])
            i0 = int(row[1])
            i1 = int(row[2])
            depth = float(row[3])

            curves_dict[curve_id].append((i0, i1))
            depths_dict[curve_id] = depth  # dernière profondeur lue pour cet ID

    if not curves_dict:
        raise ValueError("Le fichier de segments est vide")

    # --- Conversion en liste indexée par ID ---
    max_id = max([cid for cid in curves_dict.keys() if cid >= 0] + [-1])
    curves = [[] for _ in range(max_id + 2)]  # dernière position pour id=-1
    depths = [None for _ in range(max_id + 2)]

    for cid, segs in curves_dict.items():
        if cid >= 0:
            curves[cid] = segs
            depths[cid] = depths_dict[cid]
        else:
            curves[-1].extend(segs)
            depths[-1] = depths_dict[cid]

    return np.array(pts), curves, depths



def read_points (INPUT_CSV_POINTS):
    """
    Lit un fichier CSV contenant un nuage de points 3D.

    Parameters
    ----------
    INPUT_CSV_POINTS : str
        Chemin vers le fichier CSV.
        Le fichier doit contenir les colonnes : "x", "y", "z".

    Returns
    -------
    points : np.ndarray
        Tableau de shape (N, 3) contenant les coordonnées des points sous forme (x, y, z).

    Raises
    ------
    KeyError
        Si les colonnes "x", "y" ou "z" sont absentes.
    pandas.errors.EmptyDataError
        Si le fichier est vide.

    Notes
    -----
    - La lecture est effectuée via pandas.read_csv.
    - L'ordre des colonnes est explicitement imposé : x, y, z.
    """

    data = pd.read_csv(INPUT_CSV_POINTS)
    points = data[["x", "y", "z"]].values
    return points

# ------------------------
# ÉCRITURE CSV
# ------------------------
def save_CSV(OUTPUT_POINTS, points_list, OUTPUT_LINES = None, lines_list=None, save_lines=False):
    """
    Sauvegarde des points 3D (et optionnellement des segments) au format CSV.

    Parameters
    ----------
    OUTPUT_POINTS : str
        Chemin du fichier CSV de sortie pour les points.
    points_list : iterable
        Liste ou tableau contenant les points sous forme (x, y, z).
    OUTPUT_LINES : str, optional
        Chemin du fichier CSV de sortie pour les segments.
    lines_list : iterable, optional
        Liste des segments sous forme (id, ix, iy, depth).
    save_lines : bool, default=False
        Si True, les segments sont également sauvegardés.

    Returns
    -------
    None
        Génère un ou deux fichiers CSV selon la configuration.

    Raises
    ------
    ValueError
        Si save_lines=True mais que OUTPUT_LINES ou lines_list ne sont pas fournis.

    Notes
    -----
    - Les en-têtes écrits sont :
        Points : ["x", "y", "z"]
        Lignes : ["id", "ix", "iy", "depth"]
    - Les fichiers sont écrasés s'ils existent déjà.
    """

    # sauvegarde des points
    with open(OUTPUT_POINTS, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "z"])
        writer.writerows(points_list)

    # sauvegarde optionnelle des lignes
    if save_lines:
        if OUTPUT_LINES is None or lines_list is None:
            raise ValueError("OUTPUT_LINES et lines_list doivent être fournis si save_lines=True")

        with open(OUTPUT_LINES, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "ix", "iy", "depth"])
            writer.writerows(lines_list)


def load_obj_for_dico(path):
    """
    Charge un fichier OBJ triangulaire et calcule des métriques géométriques.

    Cette fonction :
        - extrait les sommets et les faces triangulaires,
        - calcule les longueurs des arêtes,
        - calcule l'aire de chaque triangle.

    Parameters
    ----------
    path : str
        Chemin vers le fichier OBJ.
        Le fichier doit contenir :
            - des lignes "v x y z" pour les sommets,
            - des lignes "f i j k" pour les faces triangulaires.

    Returns
    -------
    dict
        Dictionnaire contenant :
            "vertices" : np.ndarray (N, 3)
                Coordonnées des sommets.
            "faces" : np.ndarray (M, 3)
                Indices des triangles (indexation 0).
            "edges_lengths" : np.ndarray (M, 3)
                Longueurs des trois arêtes de chaque triangle.
            "areas" : np.ndarray (M,)
                Aire de chaque triangle.

    Raises
    ------
    FileNotFoundError
        Si le fichier OBJ n'existe pas.
    IndexError
        Si le format des faces est incorrect.

    Notes
    -----
    - Les indices OBJ commencent à 1 ; ils sont convertis en indexation Python (0-based).
    - Les faces sont supposées triangulaires.
    - Les longueurs sont calculées via la norme euclidienne.
    - L'aire est calculée via la norme du produit vectoriel : 0.5 * || (v1 - v0) x (v2 - v0) ||.
    """

    vertices = []
    faces = []

    # --- Lecture du fichier OBJ ---
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                vertices.append([float(parts[1]),
                                 float(parts[2]),
                                 float(parts[3])])

            elif line.startswith('f '):
                parts = line.strip().split()
                # Gestion des cas type f v/vt/vn
                face = []
                for p in parts[1:4]:  # on suppose triangulaire
                    idx = p.split('/')[0]
                    face.append(int(idx) - 1)  # OBJ index start at 1
                faces.append(face)

    vertices = np.array(vertices)
    faces = np.array(faces)

    # --- Extraction des sommets des triangles ---
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    # --- Longueurs d'arêtes ---
    e0 = np.linalg.norm(v1 - v0, axis=1)
    e1 = np.linalg.norm(v2 - v1, axis=1)
    e2 = np.linalg.norm(v0 - v2, axis=1)

    edges_lengths = np.vstack((e0, e1, e2)).T

    # --- Aire des triangles ---
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)

    return {
        "vertices": vertices,
        "faces": faces,
        "edges_lengths": edges_lengths,
        "areas": areas
    }
