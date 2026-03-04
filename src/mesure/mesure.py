#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script d'analyse de maillages 3D.

Ce module permet de calculer différentes métriques de qualité sur des surfaces triangulées (format .obj) et de générer des histogrammes
comparatifs :

    - aspect ratio des triangles,
    - mean ratio,
    - distribution des angles,
    - condition number des éléments.

Utilisation en ligne de commande
---------------------------------

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

Sortie
------

Une figure PNG contenant :
    - n lignes (une par maillage)
    - 4 colonnes correspondant aux métriques calculées

La figure est sauvegardée avec une résolution de 300 dpi.

Raises
------

FileNotFoundError
    Si un fichier spécifié n'existe pas ou si aucun .obj n'est trouvé.

NotADirectoryError
    Si le chemin fourni avec --all n'est pas un dossier valide.
"""



from ..io import RAndW
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.linalg import svd, LinAlgError



def plot_histogram(ax, data, xlabel="Valeurs", ylabel="nombre", bins=30):
    """
    Trace un histogramme 1D à partir d'un tableau de données.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Objet Axes sur lequel tracer l'histogramme.
    data : array-like
        Données à représenter. Doit être de forme (N,) ou (1, N).
    xlabel : str, optional
        Label de l'axe des abscisses. Par défaut "Valeurs".
    ylabel : str, optional
        Label de l'axe des ordonnées. Par défaut "Fréquence".
    bins : int, optional
        Nombre de classes (bins) pour l'histogramme. Par défaut 30.
    density : bool, optional
        Si True, normalise l'histogramme pour représenter une densité
        de probabilité. Par défaut False.

    Raises
    ------
    ValueError
        Si "data" n'est pas de dimension (N,) ou (1, N).

    Returns
    -------
    None
        La fonction modifie directement l'objet "ax" en y traçant l'histogramme.
    """
    data = np.asarray(data)

    # Accepte (1, N) ou (N,)
    if data.ndim == 2 and data.shape[0] == 1:
        data = data.flatten()
    elif data.ndim != 1:
        raise ValueError("Le tableau doit être de forme (N,) ou (1, N).")

    # Histogramme en fréquence (comptage)
    ax.hist(data, bins=bins, density=False, edgecolor='black')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    

def compute_triangle_angles(mesh):
    """
    Calcule les angles (en degrés) de chaque triangle du maillage à partir des longueurs de ses arêtes.

    Parameters
    ----------
    mesh : dict
        Dictionnaire contenant au minimum :
        - "edges_lengths" : np.ndarray de shape (M, 3)
          Longueurs des trois arêtes pour chacun des M triangles.
          Chaque ligne correspond à un triangle et contient (a, b, c).

    Returns
    -------
    angles : np.ndarray
        Tableau 1D contenant les angles de tous les triangles en degrés.
        La taille est (3*M,), correspondant aux angles A, B et C concaténés pour chaque triangle.

    Notes
    -----
    Les angles sont calculés via la loi des cosinus.
    Un clamp numérique avec np.clip est appliqué sur les cosinus afin d'éviter les erreurs numériques (valeurs légèrement hors [-1, 1])
    pouvant produire des NaN lors du calcul de arccos.
    """

    L = mesh["edges_lengths"]

    a = L[:, 0]
    b = L[:, 1]
    c = L[:, 2]

    # Loi des cosinus
    cosA = (b**2 + c**2 - a**2) / (2 * b * c)
    cosB = (a**2 + c**2 - b**2) / (2 * a * c)
    cosC = (a**2 + b**2 - c**2) / (2 * a * b)

    # Clamp numérique (évite NaN si léger dépassement)
    cosA = np.clip(cosA, -1.0, 1.0)
    cosB = np.clip(cosB, -1.0, 1.0)
    cosC = np.clip(cosC, -1.0, 1.0)

    A = np.arccos(cosA) * 180/np.pi
    B = np.arccos(cosB) * 180/np.pi
    C = np.arccos(cosC) * 180/np.pi

    angles = np.hstack((A, B, C))

    return angles


def compute_element_condition_number(mesh):
    """
    Calcule le condition number pour chaque triangle du maillage.

    Parameters
    ----------
    mesh : dict
        Dictionnaire renvoyé par "load_obj_for_dico", contenant au minimum :
        - "vertices" : np.ndarray de shape (N, 3)
        - "faces" : np.ndarray de shape (M, 3)

    Returns
    -------
    Ka : np.ndarray
        Tableau des condition numbers pour chaque triangle.
        Si le SVD ne converge pas, la valeur est np.nan.
    """
    vertices = mesh["vertices"]
    faces = mesh["faces"]
    Ka = []

    for face in faces:
        try:
            # Coordonnées des 3 sommets du triangle
            v0 = vertices[face[0]]
            v1 = vertices[face[1]]
            v2 = vertices[face[2]]

            # Matrice de transformation pour le triangle (2 vecteurs côtés)
            T = np.array([v1 - v0, v2 - v0]).T  # shape (3,2)

            # Calcul du SVD
            try:
                U, s, Vh = svd(T, full_matrices=False, check_finite=True)
                cond_num = s[0] / s[-1] if s[-1] > 0 else np.inf
            except LinAlgError:
                cond_num = np.nan

            Ka.append(cond_num)

        except Exception:
            Ka.append(np.nan)

    return np.array(Ka)


def compare_mesure_simu (lst_path, output_path):
    """
    Compare plusieurs maillages en visualisant différentes métriques géométriques sous forme d'histogrammes.

    Pour chaque maillage fourni, la fonction calcule et affiche :
        - l'aspect ratio des triangles,
        - le mean ratio,
        - les angles des triangles (en degrés),
        - le condition number des éléments K(a).

    Les résultats sont organisés sous forme d'une figure comportant n lignes (une par maillage) et 4 colonnes (une par métrique).

    Parameters
    ----------
    lst_path : list of str
        Liste des chemins vers les fichiers maillage (format compatible avec "RAndW.load_obj_for_dico").
    output_path : str
        Chemin du fichier image de sortie (ex: ".png", ".jpg", ".pdf").

    Returns
    -------
    None
        La fonction sauvegarde directement la figure à l'emplacement "output_path" avec une résolution de 300 dpi, puis ferme la figure.

    Notes
    -----
    - L'aspect ratio est défini comme :     max(arêtes) / min(arêtes) pour chaque triangle.
    - Le mean ratio est calculé via :       (4 * sqrt(3) * aire) / somme(des longueurs^2)
    - Les angles sont calculés à l'aide de "compute_triangle_angles".
    - Le condition number K(a) est calculé via "compute_element_condition_number".
    - Certaines lignes verticales de référence sont ajoutées pour visualiser des seuils de qualité (ex: angles critiques, aspect ratio élevé, etc.).
    - La taille de la police des titres de lignes est automatiquement ajustée en fonction du nombre de maillages comparés.
    """
    
    n = len(lst_path)
    col_titles = ["aspect ratio", "mean ratio", "K(a)", "angles", "aires"]

    fig, axs = plt.subplots(n, len(col_titles), figsize=(16, 3*n), squeeze=False)
   
    # --- Ajuster les marges ---
    plt.subplots_adjust(
        left=0.08,   # espace gauche (0=bord de la figure)
        right=0.97,  # espace droit
        top=0.85,    # espace haut
        bottom=0.10, # espace bas
        hspace=0.3,  # espace vertical entre subplots
        wspace=0.35  # espace horizontal entre subplots
    )



    for i in range(n):
        mesh = RAndW.load_obj_for_dico(lst_path[i])

        # aspect_ratio
        ax = axs[i,0]
        aspect_ratio = np.max(mesh["edges_lengths"], axis=1) / np.min(mesh["edges_lengths"], axis=1)
        plot_histogram(ax, aspect_ratio, "valeurs", "nombre")
        ax.axvline(x=5, color='red', linestyle='--', linewidth=2)


        # mean_ratio
        ax = axs[i,1]
        l2_sum = np.sum(mesh["edges_lengths"]**2, axis=1)
        mean_ratio = (4 * np.sqrt(3) * mesh["areas"]) / l2_sum
        plot_histogram(ax, mean_ratio, "valeurs", "nombre")


        # Ka
        ax = axs[i,2]
        try:
            Ka = compute_element_condition_number(mesh)
        except LinAlgError:
            print(f"Warning: SVD did not converge pour {lst_path[i]}. Valeurs ignorées.")
            Ka = np.array([])  # ou une valeur par défaut

        max_val = np.max(Ka)
        Ka = Ka[Ka != max_val]
        plot_histogram(ax, Ka, "valeurs", "nombre")
        ax.axvline(x=10, color='red', linestyle='--', linewidth=2)


        # angles
        ax = axs[i,3]
        angles = compute_triangle_angles(mesh)
        plot_histogram(ax, angles, "valeurs", "nombre")
        ax.axvline(x=20, color='red', linestyle='--', linewidth=2)
        ax.axvline(x=25, color='green', linestyle='--', linewidth=2)
        ax.axvline(x=120, color='green', linestyle='--', linewidth=2)
        ax.axvline(x=150, color='red', linestyle='--', linewidth=2)


        # aires
        ax = axs[i,4]
        plot_histogram(ax, mesh["areas"], "valeurs", "nombre")




    # --- Titres des colonnes ---
    for j in range(len(col_titles)):
        axs[0, j].set_title(col_titles[j], fontsize=12, pad=15)

    # --- Titres des lignes ---
    row_titles = [os.path.basename(path) for path in lst_path]

    max_fontsize = 12    # police max si peu de lignes
    min_fontsize = 6     # police minimale si beaucoup de lignes
    fontsize = max(min_fontsize, max_fontsize - (n - 1) * 0.5)



    for i, title in enumerate(row_titles):
        bbox = axs[i,0].get_position()  # position [x0, y0, width, height] en figure fraction
        y_pos = bbox.y0 + bbox.height/2  # centre vertical de la ligne i

        fig.text(0.04, y_pos, title, rotation=90, va='center', ha='right', fontsize=fontsize)
    
    plt.savefig(output_path, dpi=300)
    plt.close(fig)




if __name__ == "__main__":
    import argparse
    import glob

    # ----------------------------------------------------------------
    # ARGUMENTS CLI
    # ----------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Calcule différentes mesures pour des surfaces 3D et génère des histogrammes."
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--name",
        type=str,
        help="Chemin vers un seul fichier .obj à traiter"
    )
    group.add_argument(
        "--all",
        type=str,
        help="Chemin vers un dossier contenant des fichiers .obj à traiter"
    )

    parser.add_argument(
        "--o",
        type=str,
        default=None,
        help="Nom du fichier de sortie (PNG). Par défaut : <fichier/dossier>_mesure_simu.png"
    )

    args = parser.parse_args()

    # ----------------------------------------------------------------
    # LISTE DES FICHIERS À TRAITER
    # ----------------------------------------------------------------
    if args.name:
        if not os.path.isfile(args.name):
            raise FileNotFoundError(f"Le fichier {args.name} n'existe pas.")
        lst_path = [args.name]
        default_output = args.name.replace(".obj", "_mesure_simu.png")

    elif args.all:
        if not os.path.isdir(args.all):
            raise NotADirectoryError(f"Le dossier {args.all} n'existe pas.")
        lst_path = sorted(glob.glob(os.path.join(args.all, "*.obj")))
        if not lst_path:
            raise FileNotFoundError(f"Aucun fichier .obj trouvé dans {args.all}")
        default_output = f"{args.all}/{os.path.basename(os.path.normpath(args.all))}_mesure_simu.png"

    # ----------------------------------------------------------------
    # FICHIER DE SORTIE
    # ----------------------------------------------------------------
    output_path = args.o if args.o else default_output

    # ----------------------------------------------------------------
    # APPLICATION
    # ----------------------------------------------------------------
    print(output_path)
    compare_mesure_simu(lst_path, output_path)
    print(f"Figure sauvegardée dans : {output_path}")


