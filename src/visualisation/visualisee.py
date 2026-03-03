#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script d'analyse de maillages 3D.

Ce module permet de calculer différentes métriques de qualité sur des surfaces triangulées (format .obj) et de générer des histogrammes comparatifs :

    - aspect ratio des triangles,
    - mean ratio,
    - distribution des angles,
    - condition number des éléments.

Les figures générées sont sauvegardées automatiquement au format PNG.

Utilisation en ligne de commande
--------------------------------

Traitement d'un seul fichier :

    python script.py --name chemin/vers/maillage.obj

Traitement de tous les fichiers .obj d'un dossier :

    python script.py --all chemin/vers/dossier/

Optionnel :

    --o nom_fichier.png
        Définit le nom du fichier image de sortie.
        Par défaut :
            - <fichier>_mesure_simu.png
            - <dossier>/<dossier>_mesure_simu.png

Logique des options
-------------------

- Si aucune option (--name ou --all) n'est fournie :
       une erreur est levée (fichier ou dossier requis).
- Si --name est fourni :
       traitement du fichier unique.
- Si --all est fourni :
       traitement de tous les fichiers .obj du dossier.

Organisation des fichiers
--------------------------

Entrées :
    - Fichier .obj unique : chemin spécifié via --name
    - Dossier de fichiers .obj : chemin spécifié via --all

Sortie :
    - Une figure PNG contenant :
        - n lignes (une par maillage)
        - 4 colonnes correspondant aux métriques calculées
    - La figure est sauvegardée avec une résolution de 300 dpi.

Les dossiers de sortie sont créés automatiquement si nécessaires.

Gestion des erreurs
-------------------

FileNotFoundError
    Si un fichier spécifié n'existe pas ou si aucun .obj n'est trouvé.

NotADirectoryError
    Si le chemin fourni avec --all n'est pas un dossier valide.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from ..io import RAndW


# =============================================================================================================
# VISUALISATION
# =============================================================================================================

def plot_point_cloud(points, output_path):
    """
    Affiche et sauvegarde un nuage de points 3D.

    Parameters
    ----------
    points : np.ndarray
        Tableau de shape (N, 3) contenant les coordonnées des points.
        Chaque ligne correspond à un point (x, y, z).
    output_path : str
        Chemin du fichier image de sortie (ex: ".png", ".jpg", ".pdf").

    Returns
    -------
    None
        La fonction génère une figure 3D représentant le nuage de points,
        la sauvegarde à l'emplacement "output_path" avec une résolution
        de 300 dpi, puis ferme la figure.

    Notes
    -----
    - La couleur des points est déterminée par leur coordonnée en z.
    - La colormap utilisée est "viridis".
    - Les points sont affichés avec un marqueur "x" et une taille réduite (s=1) afin d'améliorer la lisibilité pour de grands nuages.
    """

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               c=points[:, 2], cmap="viridis", s=1, marker="x")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Nuage de points 3D")

    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_curves(pts, curves, depths, output_path):
    """
    Trace et sauvegarde des courbes 3D générées par un algorithme de type Marching Squares.

    Parameters
    ----------
    pts : np.ndarray
        Tableau de shape (N, 3) contenant les coordonnées des points.
        Les indices référencés dans curves correspondent aux lignes de ce tableau.
    curves : list of list of tuple(int, int)
        Liste de courbes. Chaque courbe est une liste de segments, et chaque segment est défini par un tuple (i0, i1) indiquant
        les indices des deux points dans pts.
    depths : list of float or None
        Liste des profondeurs associées à chaque courbe.
        Si une profondeur vaut None, la courbe correspondante n'est pas tracée.
    output_path : str
        Chemin du fichier image de sortie (ex: ".png", ".jpg", ".pdf").

    Raises
    ------
    ValueError
        Si aucune profondeur valide (non None) n'est disponible pour la normalisation des couleurs.

    Returns
    -------
    None
        La fonction génère une figure 3D contenant les courbes,
        la sauvegarde à l'emplacement output_path avec une résolution de 300 dpi, puis ferme la figure.

    Notes
    -----
    - Les courbes sont tracées dans le plan (X, Y) à une altitude Z correspondant à leur profondeur.
    - La couleur de chaque courbe est déterminée par sa profondeur, normalisée entre la valeur minimale et maximale des profondeurs
      valides.
    - La colormap utilisée est "viridis".
    - Une vue en plan peut être activée en décommentant :
          ax.view_init(elev=90, azim=0)
    """

    # Filtrer les profondeurs valides
    valid_depths = [d for d in depths if d is not None]
    if not valid_depths:
        raise ValueError("Aucune profondeur valide à tracer")

    norm = colors.Normalize(vmin=min(valid_depths), vmax=max(valid_depths))
    cmap = cm.viridis

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Parcours des courbes
    for curve_id, segments in enumerate(curves):

        if not segments:
            continue

        depth = depths[curve_id]
        if depth is None:
            continue

        color = cmap(norm(depth))

        # Tracé de chaque segment de la courbe
        for i0, i1 in segments:
            ax.plot(
                [pts[i0, 0], pts[i1, 0]],
                [pts[i0, 1], pts[i1, 1]],
                [depth, depth],
                color=color,
                linewidth=1
            )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Courbes 3D — Marching Squares")

    # Vue optionnelle en plan
    # ax.view_init(elev=90, azim=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


# =============================================================================================================
# MAIN
# =============================================================================================================
def main_visualise( name="Lake_Aydat", show_points=True, show_curves=True):
    """
    Visualise un nuage de points et/ou des courbes 3D associées à un jeu de données donné.

    Parameters
    ----------
    name : str, optional
        Nom du jeu de données (sans extension). Ce nom est utilisé
        pour construire automatiquement les chemins d'accès :
            - ../point_cloud/{name}.csv
            - ../curves/{name}_points.csv
            - ../curves/{name}_lines.csv
        Par défaut : "Lake_Aydat".
    show_points : bool, optional
        Si True, affiche le nuage de points 3D. Par défaut True.
    show_curves : bool, optional
        Si True, affiche les courbes 3D issues du Marching Squares.
        Par défaut True.

    Returns
    -------
    figures : dict
        Dictionnaire contenant les figures matplotlib générées :
            - "points" : figure du nuage de points (ou None)
            - "curves" : figure des courbes 3D (ou None)

    Notes
    -----
    - Le nuage de points est coloré selon la coordonnée Z (profondeur) avec la colormap "viridis".
    - Les courbes sont tracées dans le plan (X, Y) à une altitude Z correspondant à leur profondeur.
    - Les couleurs des courbes sont normalisées entre la profondeur minimale et maximale.
    - Les figures ne sont pas sauvegardées automatiquement ; elles sont retournées afin de permettre une manipulation ou une
      sauvegarde ultérieure.
    """

    figures = {
        "points": None,
        "curves": None
    }

    # chemins d'entrée
    cloud_csv = f"../point_cloud/{name}.csv"
    curves_dir = "../curves/"
    curves_points_csv = os.path.join(curves_dir, f"{name}_points.csv")
    curves_lines_csv = os.path.join(curves_dir, f"{name}_lines.csv")

    # ----------------------------------------------------------------
    # NUAGE DE POINTS
    # ----------------------------------------------------------------
    if show_points:
        points = RAndW.read_points(cloud_csv)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            c=points[:, 2],
            cmap="viridis",
            s=1,
            marker="x"
        )

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (profondeur)")
        ax.set_title("Nuage de points 3D")

        figures["points"] = fig

    # ----------------------------------------------------------------
    # COURBES
    # ----------------------------------------------------------------
    if show_curves:
        pts, lines, depths = RAndW.read_curves(
            curves_points_csv,
            curves_lines_csv
        )

        norm = colors.Normalize(vmin=min(depths), vmax=max(depths))
        cmap = cm.viridis

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")

        for (i, j), d in zip(lines, depths):
            ax.plot(
                [pts[i, 0], pts[j, 0]],
                [pts[i, 1], pts[j, 1]],
                [d, d],
                color=cmap(norm(d)),
                linewidth=1
            )

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (profondeur)")
        ax.set_title("Courbes 3D — Marching Squares")

        figures["curves"] = fig

    return figures



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualisation 3D d'un nuage de points et/ou de courbes."
    )

    parser.add_argument(
        "--name",
        type=str,
        default="Lake_Aydat",
        help="Nom du jeu de données (sans extension)."
    )

    parser.add_argument(
        "--p",
        action="store_true",
        help="Afficher uniquement le nuage de points."
    )

    parser.add_argument(
        "--c",
        action="store_true",
        help="Afficher uniquement les courbes."
    )

    parser.add_argument(
        "--o",
        "--output",
        type=str,
        default=None,
        help="Dossier de sortie des images."
    )

    args = parser.parse_args()

    # logique p / c
    do_points = args.p or not args.c
    do_curves = args.c or not args.p

    name = args.name

    # chemins
    cloud_csv = f"data/processed/point_cloud/{name}.csv"
    curves_dir = "data/processed/curves/"
    curves_points_csv = os.path.join(curves_dir, f"{name}_points.csv")
    curves_lines_csv = os.path.join(curves_dir, f"{name}_lines.csv")

    output_dir = args.o if args.o else f"data/final/{name}/"
    os.makedirs(output_dir, exist_ok=True)

    try:
        if do_points:
            print("📊 Nuage de points...")
            points = RAndW.read_points(cloud_csv)
            out_img = os.path.join(output_dir, f"{name}_nuage_points.png")
            plot_point_cloud(points, out_img)
            print(f"✅ Nuage enregistré : {out_img}")

        if do_curves:
            print("📈 Courbes...")
            pts, lines, depths = RAndW.read_curves(curves_points_csv, curves_lines_csv)
            out_img = os.path.join(output_dir, f"{name}_curves.png")
            plot_curves(pts, lines, depths, out_img)
            print(f"✅ Courbes enregistrées : {out_img}")

    except Exception as e:
        print("\n❌ ERREUR :", e)
        sys.exit(1)