#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script de génération d'un maillage 3D via Constrained Delaunay Triangulation (CDT).

Ce module permet de créer un maillage triangulé à partir de points 3D et de segments de courbes (PSLG), puis de sauvegarder le résultat au format OBJ.

Utilisation en ligne de commande
--------------------------------

Exécution par défaut avec le jeu de données Lake_Aydat :

    python script_CDT.py

Spécifier un autre jeu de données :

    python script_CDT.py --name NomDuJeu

Spécifier un chemin de sortie personnalisé pour le fichier OBJ :

    python script_CDT.py --o data/final/NomDuJeu/mon_maillage.obj

Logique des options
-------------------

- --name : nom du jeu de données (sans extension), utilisé pour construire automatiquement les chemins vers les fichiers CSV d'entrée.

- --o ou --output : chemin complet du fichier OBJ de sortie.
  Si non fourni, le fichier est créé dans "data/final/{name}/{name}_CDT.obj".

Organisation des fichiers
--------------------------

Entrées (doivent exister) :
    data/processed/curves/{name}_points.csv
    data/processed/curves/{name}_lines.csv

Sortie :
    data/final/{name}/{name}_CDT.obj  (ou chemin personnalisé via --o)

Pipeline exécuté
----------------

1. Lecture des points et segments via "read_points_and_segments".
2. Vérification qu'au moins 3 points existent.
3. Triangulation CDT via "applique_CDT".
4. Interpolation des coordonnées z sur les nouveaux sommets via "interpolate_z".
5. Sauvegarde du maillage final au format OBJ via "save_obj_CDT".

Gestion des erreurs
-------------------

- Si un fichier d'entrée est introuvable, une exception "FileNotFoundError" est levée.
- Si le nombre de points est inférieur à 3, une exception "ValueError" est levée.
- Toute autre exception est capturée, un message est affiché et le programme s'arrête avec un code de sortie 1.

Notes
-----

- Les triangles générés respectent les segments imposés (PSLG).
- Les nouveaux sommets ajoutés lors du raffinement reçoivent une valeur z interpolée.
- Le script peut être appelé depuis d'autres programmes ou intégré dans un pipeline automatisé.
"""

import os
import sys
import numpy as np
import pandas as pd
import triangle as tr
from scipy.spatial import Delaunay

from ..io import RAndW


# =============================================================================================================
# PARAMÈTRES
# =============================================================================================================
def determine_param(name, output_override=None):
    """
    Détermine les chemins d'entrée et de sortie pour la génération d'un maillage CDT (Constrained Delaunay Triangulation).

    Parameters
    ----------
    name : str
        Nom du jeu de données (sans extension). Utilisé pour construire automatiquement les chemins d'entrée :
            - data/processed/curves/{name}_points.csv
            - data/processed/curves/{name}_lines.csv
    output_override : str or None, optional
        Chemin complet du fichier .obj de sortie.
        Si fourni, ce chemin est utilisé directement et son dossier parent est créé si nécessaire.
        Si None, le fichier de sortie est généré automatiquement dans :
            data/final/{name}/{name}_CDT.obj

    Returns
    -------
    POINTS_CSV : str
        Chemin vers le fichier CSV contenant les points des courbes.
    LINES_CSV : str
        Chemin vers le fichier CSV contenant les segments des courbes.
    OUTPUT_OBJ : str
        Chemin vers le fichier .obj de sortie pour le maillage CDT.

    Notes
    -----
    - Les dossiers de sortie sont créés automatiquement s'ils n'existent pas.
    - Cette fonction ne vérifie pas l'existence des fichiers d'entrée.
    - Elle centralise la logique des chemins afin d'éviter la duplication dans le reste du programme.
    """
        
    POINTS_CSV = f"data/processed/curves/{name}_points.csv"
    LINES_CSV  = f"data/processed/curves/{name}_lines.csv"

    if output_override is not None:
        OUTPUT_OBJ = output_override
        output_dir = os.path.dirname(OUTPUT_OBJ)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = f"data/final/{name}/"
        os.makedirs(output_dir, exist_ok=True)
        OUTPUT_OBJ = os.path.join(output_dir, f"{name}_CDT.obj")

    return POINTS_CSV, LINES_CSV, OUTPUT_OBJ


# =============================================================================================================
# LECTURE DES DONNÉES
# =============================================================================================================
def read_points_and_segments(points_csv, lines_csv):
    """
    Lit les points et segments de courbes, puis prépare les données pour une triangulation CDT (Constrained Delaunay Triangulation).

    Parameters
    ----------
    points_csv : str
        Chemin vers le fichier CSV contenant les points des courbes.
    lines_csv : str
        Chemin vers le fichier CSV contenant les segments des courbes.

    Returns
    -------
    points_xy : np.ndarray
        Tableau de shape (N, 2) contenant les coordonnées (x, y) des points.
    z_values : np.ndarray
        Tableau de shape (N,) contenant les coordonnées z associées aux points.
    segments_array : np.ndarray
        Tableau de shape (M, 2) contenant les indices des segments (paires d'indices de points) à imposer dans la triangulation.
        Si aucun segment n'est présent, retourne un tableau vide de shape (0, 2).

    Notes
    -----
    - Les données sont lues via "RAndW.read_curves".
    - Toutes les courbes sont fusionnées en une liste unique de segments afin de construire un PSLG (Planar Straight Line Graph).
    - La triangulation sera réalisée uniquement sur les coordonnées (x, y).
    """

    pts, curves, depths = RAndW.read_curves(points_csv, lines_csv)

    points_xy = pts[:, :2]
    z_values = pts[:, 2]

    # Fusionner toutes les courbes en segments pour CDT
    segments_list = []
    for curve in curves:
        segments_list.extend(curve)

    if not segments_list:
        segments_array = np.empty((0, 2), dtype=int)
    else:
        segments_array = np.array(segments_list, dtype=int)

    return points_xy, z_values, segments_array


# =============================================================================================================
# CDT
# =============================================================================================================
def applique_CDT(points_xy, segments):
    """
    Applique une triangulation de Delaunay contrainte (CDT) à un ensemble de points 2D et de segments imposés.

    Parameters
    ----------
    points_xy : np.ndarray
        Tableau de shape (N, 2) contenant les coordonnées (x, y) des sommets.
    segments : np.ndarray
        Tableau de shape (M, 2) contenant les indices des segments à respecter dans la triangulation (PSLG).

    Returns
    -------
    vertices : np.ndarray
        Tableau de shape (K, 2) contenant les sommets de la triangulation générée (peut inclure des points ajoutés lors du raffinement).
    triangles : np.ndarray
        Tableau de shape (T, 3) contenant les indices des triangles formant la triangulation.

    Notes
    -----
    La triangulation est réalisée via "triangle.triangulate" avec les options :

        - "p"  : respect des segments (PSLG)
        - "q30": angle minimal de 30 degrés
        - "a1000": aire maximale des triangles fixée à 1000
        - "D"  : raffinement de Delaunay

    Ces paramètres permettent d'améliorer la qualité du maillage en contrôlant les angles et la taille des éléments.
    """

    A = dict(vertices=points_xy, segments=segments)

    # p  : PSLG (respect des segments)
    # q30: angle minimal 30°
    # a1000 : aire maximale
    # D  : Delaunay refinement
    B = tr.triangulate(A, "pq30a1000D")

    return B["vertices"], B["triangles"]


# =============================================================================================================
# INTERPOLATION Z
# =============================================================================================================
def interpolate_z(original_xy, original_z, new_vertices):
    """
    Interpole les coordonnées z sur de nouveaux sommets à partir d'un nuage de points 2D original.

    Parameters
    ----------
    original_xy : np.ndarray
        Tableau de shape (N, 2) contenant les coordonnées (x, y) des points originaux.
    original_z : np.ndarray
        Tableau de shape (N,) contenant les valeurs z associées aux points originaux.
    new_vertices : np.ndarray
        Tableau de shape (M, 2) contenant les coordonnées (x, y) des nouveaux sommets (issus d'une triangulation CDT).

    Returns
    -------
    new_z : np.ndarray
        Tableau de shape (M,) contenant les valeurs z interpolées pour tous les sommets. Les points originaux conservent
        leurs z initiaux, les nouveaux points reçoivent une valeur interpolée via coordonnées barycentriques si dans un triangle
        existant, sinon la valeur z du point original le plus proche.

    Notes
    -----
    - La triangulation Delaunay est utilisée pour déterminer dans quel triangle chaque nouveau point se trouve.
    - Les coordonnées barycentriques sont calculées pour interpoler la valeur z.
    - Si un point se trouve en dehors du domaine, la valeur z du point original le plus proche est utilisée.
    """

    new_z = np.zeros(len(new_vertices))
    new_z[:len(original_z)] = original_z

    tri_orig = Delaunay(original_xy)

    for i in range(len(original_z), len(new_vertices)):
        pt = new_vertices[i]
        simplex = tri_orig.find_simplex(pt)

        if simplex >= 0:
            verts = tri_orig.simplices[simplex]
            transform = tri_orig.transform[simplex]
            bary = np.dot(transform[:2], pt - transform[2])
            bary = np.append(bary, 1 - bary.sum())
            new_z[i] = np.dot(bary, original_z[verts])
        else:
            dists = np.linalg.norm(original_xy - pt, axis=1)
            new_z[i] = original_z[np.argmin(dists)]

    return new_z


# =============================================================================================================
# ÉCRITURE OBJ
# =============================================================================================================
def save_obj_CDT(output_obj, vertices, triangles, z_values):
    """
    Sauvegarde une triangulation CDT au format OBJ 3D.

    Parameters
    ----------
    output_obj : str
        Chemin du fichier OBJ de sortie.
    vertices : np.ndarray
        Tableau de shape (N, 2) contenant les coordonnées (x, y) des sommets de la triangulation.
    triangles : np.ndarray
        Tableau de shape (T, 3) contenant les indices des sommets formant chaque triangle.
    z_values : np.ndarray
        Tableau de shape (N,) contenant les valeurs z associées à chaque sommet.

    Returns
    -------
    None
        La fonction écrit directement un fichier OBJ, où :
        - chaque ligne "v x y z" correspond à un sommet
        - chaque ligne "f i j k" correspond à un triangle (indices 1-based)

    Notes
    -----
    - Les indices des faces dans le fichier OBJ commencent à 1 conformément au format OBJ.
    - Cette fonction ne vérifie pas la validité des tableaux fournis ; il est supposé que vertices, triangles et z_values
      ont des dimensions compatibles.
    """

    with open(output_obj, "w") as f:

        for i, v in enumerate(vertices):
            f.write(f"v {v[0]} {v[1]} {z_values[i]}\n")

        for tri in triangles:
            i, j, k = tri + 1
            f.write(f"f {i} {j} {k}\n")


# =============================================================================================================
# MAIN
# =============================================================================================================
def CDT_main(POINTS_CSV, LINES_CSV, OUTPUT_OBJ):
    """
    Exécute le pipeline complet pour générer un maillage CDT 3D à partir de points et de segments de courbes et sauvegarde
    le résultat au format OBJ.

    Parameters
    ----------
    POINTS_CSV : str
        Chemin vers le fichier CSV contenant les points (x, y, z) des courbes.
    LINES_CSV : str
        Chemin vers le fichier CSV contenant les segments (indices des points) des courbes.
    OUTPUT_OBJ : str
        Chemin du fichier OBJ de sortie pour le maillage généré.

    Raises
    ------
    FileNotFoundError
        Si les fichiers POINTS_CSV ou LINES_CSV n'existent pas.
    ValueError
        Si le nombre de points est inférieur à 3 (impossible de former une triangulation).

    Returns
    -------
    None

    Notes
    -----
    Le pipeline exécuté est le suivant :
        1. Lecture des points et segments via "read_points_and_segments".
        2. Vérification qu'au moins 3 points existent.
        3. Triangulation CDT via "applique_CDT".
        4. Interpolation des valeurs z sur les nouveaux sommets via "interpolate_z".
        5. Sauvegarde du maillage final au format OBJ via "save_obj_CDT".

    - Les triangles générés respectent les segments imposés (PSLG).
    - Les nouveaux sommets ajoutés lors du raffinement reçoivent une valeur z interpolée.
    """
     
    if not os.path.isfile(POINTS_CSV):
        raise FileNotFoundError(f"Fichier points introuvable : {POINTS_CSV}")

    if not os.path.isfile(LINES_CSV):
        raise FileNotFoundError(f"Fichier segments introuvable : {LINES_CSV}")

    points_xy, z_values, segments = read_points_and_segments(POINTS_CSV, LINES_CSV)

    if len(points_xy) < 3:
        raise ValueError("Au moins 3 points sont nécessaires")

    vertices, triangles = applique_CDT(points_xy, segments)

    new_z = interpolate_z(points_xy, z_values, vertices)

    save_obj_CDT(OUTPUT_OBJ, vertices, triangles, new_z)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Application d'une Constrained Delaunay Triangulation (CDT)."
    )

    parser.add_argument(
        "--name",
        type=str,
        default="Lake_Aydat",
        help="Nom du jeu de données (sans extension)."
    )

    parser.add_argument(
        "--o",
        "--output",
        type=str,
        default=None,
        help="Chemin du fichier OBJ de sortie."
    )

    args = parser.parse_args()

    print("Utilisation de CDT :")

    try:
        POINTS_CSV, LINES_CSV, OUTPUT_OBJ = determine_param(args.name, args.o)

        if not os.path.isfile(POINTS_CSV):
            raise FileNotFoundError(f"Fichier points introuvable : {POINTS_CSV}")

        if not os.path.isfile(LINES_CSV):
            raise FileNotFoundError(f"Fichier segments introuvable : {LINES_CSV}")

        print("\t📊 Lecture des points et courbes")
        points_xy, z_values, segments = read_points_and_segments(POINTS_CSV, LINES_CSV)

        if len(points_xy) < 3:
            raise ValueError("Au moins 3 points sont nécessaires")

        print("\t🔺 Application du CDT")
        vertices, triangles = applique_CDT(points_xy, segments)

        print("\t📐 Interpolation des Z")
        new_z = interpolate_z(points_xy, z_values, vertices)

        print("\t💾 Sauvegarde OBJ")
        save_obj_CDT(OUTPUT_OBJ, vertices, triangles, new_z)

        print(f"\t✅ Surface CDT enregistrée : {OUTPUT_OBJ}")

    except Exception as e:
        print("\n❌ ERREUR :", e)
        sys.exit(1)