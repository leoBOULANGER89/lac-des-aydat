from ..io import RAndW
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.linalg import svd, LinAlgError



def plot_histogram_1d(ax, data, xlabel="Valeurs", ylabel="Fréquence", bins=30, density=False):
    """
    Affiche l'histogramme d'un tableau NumPy 1D (1xN ou (N,)).

    Parameters
    ----------
    data : np.ndarray
        Tableau NumPy de forme (N,) ou (1, N).
    bins : int, optional
        Nombre de classes (bins) de l’histogramme.
        Par défaut 30.
    title : str, optional
        Titre du graphique.
    xlabel : str, optional
        Label de l’axe des abscisses.
    ylabel : str, optional
        Label de l’axe des ordonnées.
    density : bool, optional
        Si True, normalise l’histogramme (aire totale = 1).
        Par défaut False.

    Raises
    ------
    ValueError
        Si le tableau n’est pas unidimensionnel.

    Returns
    -------
    None
        Affiche simplement la figure.
    """

    data = np.asarray(data)

    # Accepte (1, N) ou (N,)
    if data.ndim == 2 and data.shape[0] == 1:
        data = data.flatten()
    elif data.ndim != 1:
        raise ValueError("Le tableau doit être de forme (N,) ou (1, N).")

    ax.hist(data, bins=bins, density=density, edgecolor='black')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    

def compute_triangle_angles(mesh):
    """
    Retourne une matrice (M,1)
    contenant les 3 angles (en radians) de chaque triangle.
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
        Dictionnaire renvoyé par `load_obj_for_dico`, contenant au minimum :
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
    Compare plusieurs fichiers de maillage et affiche des histogrammes récapitulatifs.

    Pour chaque fichier de maillage dans `lst_path`, la fonction :
    1. Charge le maillage via `RAndW.load_obj_for_dico`.
    2. Calcule et affiche quatre histogrammes sur une ligne de subplots :
       - Aspect ratio des éléments (avec ligne verticale de référence à x=5)
       - Mean ratio des éléments
       - Angles des triangles (avec lignes verticales de référence à x=20, 25, 120, 150)
       - Condition number des éléments (ligne verticale à x=10, le maximum exclu)
    3. Ajoute des titres de colonnes et des titres de lignes centrés verticalement avec taille de police adaptative.
    4. Ajuste les marges et espacements pour optimiser l'utilisation de l'espace.
    5. Sauvegarde la figure finale au chemin `output_path` en haute résolution (300 dpi).

    Parameters
    ----------
    lst_path : list of str
        Liste des chemins vers les fichiers de maillage à traiter.
    output_path : str
        Chemin complet pour sauvegarder la figure générée (ex. .png, .pdf).

    Raises
    ------
    Any exception provenant du chargement des fichiers de maillage ou du calcul des histogrammes
    peut être propagée (ex. erreurs de lecture de fichier ou de forme de données).

    Returns
    -------
    None
        La fonction ne retourne rien ; elle sauvegarde simplement la figure finale.
    """
    n = len(lst_path)

    fig, axs = plt.subplots(n, 4, figsize=(16, 3*n), squeeze=False)
   
    # --- Ajuster les marges ---
    plt.subplots_adjust(
        left=0.08,   # espace gauche (0=bord de la figure)
        right=0.97,  # espace droit
        top=0.85,    # espace haut
        bottom=0.10, # espace bas
        hspace=0.4,  # espace vertical entre subplots
        wspace=0.25  # espace horizontal entre subplots
    )



    for i in range(n):
        mesh = RAndW.load_obj_for_dico(lst_path[i])


        aspect_ratio = np.max(mesh["edges_lengths"], axis=1) / np.min(mesh["edges_lengths"], axis=1)
        plot_histogram_1d(axs[i,0], aspect_ratio, "valeurs", "nombre", 30, True)
        axs[i,0].axvline(x=5, color='red', linestyle='--', linewidth=2)

        l2_sum = np.sum(mesh["edges_lengths"]**2, axis=1)
        mean_ratio = (4 * np.sqrt(3) * mesh["areas"]) / l2_sum
        plot_histogram_1d(axs[i,1], mean_ratio, "valeurs", "nombre", 30)

        angles = compute_triangle_angles(mesh)
        plot_histogram_1d(axs[i,2], angles, "valeurs", "nombre", 30)
        axs[i,2].axvline(x=20, color='red', linestyle='--', linewidth=2)
        axs[i,2].axvline(x=25, color='green', linestyle='--', linewidth=2)
        axs[i,2].axvline(x=120, color='green', linestyle='--', linewidth=2)
        axs[i,2].axvline(x=150, color='red', linestyle='--', linewidth=2)

        try:
            Ka = compute_element_condition_number(mesh)
        except LinAlgError:
            print(f"Warning: SVD did not converge pour {lst_path[i]}. Valeurs ignorées.")
            Ka = np.array([])  # ou une valeur par défaut


        Ka = compute_element_condition_number(mesh)
        max_val = np.max(Ka)
        Ka = Ka[Ka != max_val]
        plot_histogram_1d(axs[i,3], Ka, "valeurs", "nombre", 30)
        axs[i,3].axvline(x=10, color='red', linestyle='--', linewidth=2)



    # --- Titres des colonnes ---
    col_titles = ["aspect ratio", "mean ratio", "angles", "K(a)"]

    for j in range(4):
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


