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
    Vérifie que les fichiers.
    Lève FileNotFoundError si l'un des fichiers est manquant.
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
    Lit les fichiers CSV de points et de segments de courbes de niveau,
    et retourne les courbes regroupées par ID. Les segments avec le même ID,
    même s'ils ne sont pas contigus dans le CSV, sont fusionnés dans la même liste.
    Les segments avec ID = -1 (non fermés ou isolés) sont placés à la fin.

    Parameters
    ----------
    INPUT_CSV_POINTS : str
        Chemin vers le CSV contenant les points (x, y, depth).
    INPUT_CSV_LINES : str
        Chemin vers le CSV contenant les segments (id, ix, iy, depth).

    Returns
    -------
    pts : np.ndarray of shape (N, 3)
        Tableau des points [x, y, depth].
    curves : list of list of tuple
        Liste des courbes, chaque courbe est une liste de segments [(i0, i1), ...].
        L’index dans la liste correspond à l’ID de la courbe.
        Les segments avec id = -1 sont regroupés dans `curves[-1]`.
    depths : list of float
        Liste des profondeurs correspondantes à chaque courbe.
        `depths[i]` correspond à `curves[i]`.
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
    Lit un fichier CSV contenant des coordonnées 3D et retourne les points sous forme de tableau NumPy.

    Le fichier CSV doit contenir les colonnes : 'x', 'y', 'z'.
    Les valeurs sont extraites et organisées dans un tableau de forme (N, 3), où N est le nombre de points.

    Parameters
    ----------
    INPUT_CSV_POINTS : str
        Chemin vers le fichier CSV contenant le nuage de points.

    Returns
    -------
    numpy.ndarray
        Tableau de forme (N, 3) contenant les coordonnées des points sous la forme (x, y, z).

    Raises
    ------
    FileNotFoundError
        Si le fichier CSV n'existe pas.
    KeyError
        Si les colonnes 'x', 'y' ou 'z' sont absentes du fichier CSV.

    Notes
    -----
    - Le fichier CSV doit être encodé en UTF-8 pour éviter les problèmes de lecture.
    - Le type de retour est toujours un tableau NumPy de float64
    """
    data = pd.read_csv(INPUT_CSV_POINTS)
    points = data[["x", "y", "z"]].values
    return points

# ------------------------
# ÉCRITURE CSV
# ------------------------
def save_CSV(OUTPUT_POINTS, points_list, OUTPUT_LINES = None, lines_list=None, save_lines=False):
    """
    Sauvegarde des points et, optionnellement, des segments des courbes de niveau dans des fichiers CSV.

    Cette fonction écrit :
    - un fichier CSV contenant les points [x, y, z].
    - un fichier CSV contenant les segments [id, ix, iy, depth] si save_lines=True.

    Parameters
    ----------
    OUTPUT_POINTS : str
        Chemin du fichier CSV où seront enregistrés les points.
    points_list : list of list
        Liste des points sous la forme [x, y, z].
    OUTPUT_LINES : str or None, optional
        Chemin du fichier CSV où seront enregistrés les segments.
        Obligatoire si save_lines=True.
    lines_list : list of list or None, optional
        Liste des segments sous la forme [id, ix, iy, depth].
        Ignorée si save_lines=False.
    save_lines : bool, optional
        Indique si les segments doivent être sauvegardés.
        Par défaut False.

    Raises
    ------
    ValueError
        Si save_lines=True mais que OUTPUT_LINES ou lines_list n'est pas fourni.

    Returns
    -------
    None

    Notes
    -----
    - Les fichiers CSV générés contiennent des en-têtes explicites.
    - Les points et les segments sont écrits en UTF-8
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
    Charge un fichier OBJ triangulaire et calcule des informations géométriques
    utiles pour un traitement ultérieur.

    Cette fonction :
    - lit les sommets (v) et les faces triangulaires (f) d’un fichier OBJ,
    - construit les tableaux NumPy des sommets et des faces,
    - calcule les longueurs des arêtes de chaque triangle,
    - calcule l’aire de chaque triangle.

    Parameters
    ----------
    path : str
        Chemin vers le fichier OBJ à charger.
        Le fichier doit contenir des faces triangulaires (3 sommets par face).

    Returns
    -------
    dict
        Dictionnaire contenant :
        - "vertices" : np.ndarray de shape (N, 3)
            Coordonnées des sommets.
        - "faces" : np.ndarray de shape (M, 3)
            Indices des sommets formant chaque triangle (indexation 0-based).
        - "edges_lengths" : np.ndarray de shape (M, 3)
            Longueurs des trois arêtes de chaque triangle.
        - "areas" : np.ndarray de shape (M,)
            Aire de chaque triangle.

    Raises
    ------
    FileNotFoundError
        Si le fichier spécifié par `path` n'existe pas.
    ValueError
        Si le fichier contient des faces non triangulaires ou mal formatées.

    Notes
    -----
    - Les indices des faces dans le format OBJ commencent à 1 ;
      ils sont convertis en indexation 0-based pour NumPy.
    - Les normales et coordonnées de texture éventuelles (v/vt/vn)
      sont ignorées.
    - Les calculs géométriques utilisent les fonctions de `numpy.linalg`.
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
