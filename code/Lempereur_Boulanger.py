import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points

# ============================================================
# LECTURE DES CSV
# ============================================================

def read_curves(points_csv, lines_csv):
    if not os.path.isfile(points_csv):
        raise FileNotFoundError(points_csv)
    if not os.path.isfile(lines_csv):
        raise FileNotFoundError(lines_csv)

    points = []
    with open(points_csv, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            points.append((float(row[0]), float(row[1]), float(row[2])))

    segments = []
    depths = []
    with open(lines_csv, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            segments.append((int(row[0]), int(row[1])))
            depths.append(float(row[2]))

    return np.array(points), np.array(segments), np.array(depths)

# ============================================================
# CONSTRUCTION DES COURBES PAR PROFONDEUR
# ============================================================

def build_curves(points, segments, depths):
    curves = {}
    for (i, j), d in zip(segments, depths):
        p1 = points[i][:2]
        p2 = points[j][:2]
        if d not in curves:
            curves[d] = []
        curves[d].append((p1, p2))

    curves_by_depth = {}
    for d, segs in curves.items():
        coords = []
        for a, b in segs:
            if not coords:
                coords.append(a)
            coords.append(b)
        curves_by_depth[d] = LineString(coords)

    return curves_by_depth

# ============================================================
# OUTILS GEOMETRIQUES
# ============================================================

def segment_normal(a, b):
    t = b - a
    n = np.array([-t[1], t[0]])
    return n / np.linalg.norm(n)

def angle_between(u, v):
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)
    return np.arccos(np.clip(abs(np.dot(u, v)), -1.0, 1.0))

# ============================================================
# PROJECTION AVEC REPLI ANGULAIRE
# ============================================================

def project_point_with_fallback(P0, curve, prev_vector, blocking_curves, angle_max, n_samples=400):
    P_proj = np.array(nearest_points(Point(P0), curve)[1].coords[0])
    candidates = []

    cos_max = np.cos(angle_max)

    for t in np.linspace(0, 1, n_samples):
        p = np.array(curve.interpolate(t, normalized=True).coords[0])
        v = p - P0
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-9:
            continue
        v_unit = v / norm_v

        # Angle maximal avec le vecteur précédent
        if np.dot(v_unit, prev_vector) < cos_max:
            continue

        seg = LineString([P0, p])
        if any(seg.crosses(c) for c in blocking_curves):
            continue

        candidates.append((np.linalg.norm(p - P_proj), p))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]



# ============================================================
# PROJECTION ITÉRATIVE (descente puis remontée)
# ============================================================

def find_projection_iterative(P0, depth0, curves_by_depth, angle_max, max_steps=200):
    path = [(P0, depth0)]
    current_P = P0
    current_depth = depth0
    direction = "down"  # d'abord descendre
    depths_sorted = sorted(curves_by_depth.keys())

    # vecteur précédent initial : normale de la première ligne
    curve0 = curves_by_depth[depth0]
    coords0 = np.array(curve0.coords)
    i0 = np.argmin(np.linalg.norm(coords0 - P0, axis=1))
    a0 = coords0[max(i0 - 1, 0)]
    b0 = coords0[min(i0 + 1, len(coords0) - 1)]
    prev_vector = segment_normal(a0, b0)

    for _ in range(max_steps):
        # déterminer les lignes à tester
        if direction == "down":
            candidate_depths = [d for d in depths_sorted if d < current_depth]
        else:
            candidate_depths = [d for d in reversed(depths_sorted) if d > current_depth]

        found = False
        if (candidate_depths != []):
            d = candidate_depths[0]
            # projection
            P1 = project_point_with_fallback(
                current_P,
                curves_by_depth[d],
                prev_vector,
                [c for dd, c in curves_by_depth.items() if dd != d],
                angle_max
            )
            if P1 is not None:
                # mise à jour du vecteur précédent
                prev_vector = P1 - current_P
                prev_vector /= np.linalg.norm(prev_vector)

                # mise à jour du chemin
                path.append((P1, d))
                current_P = P1
                current_depth = d
                # si on a trouvé un point, on arrête de chercher parmi les candidats
                found = True
                # si on descendait et qu'on a trouvé, on continue en descendant
                # sinon on remonte naturellement à la prochaine itération

        if not found:
            # si rien n'a été trouvé, on cherche sur ça propre ligne et on inverse la direction
            direction = "up" if direction == "down" else "down"
            d = current_depth

            P1 = project_point_with_fallback(
                current_P,
                curves_by_depth[d],
                prev_vector,
                [c for dd, c in curves_by_depth.items() if dd != d],
                angle_max
            )

            if P1 is not None:
                # mise à jour du vecteur précédent
                prev_vector = P1 - current_P
                prev_vector /= np.linalg.norm(prev_vector)

                # mise à jour du chemin
                path.append((P1, d))
                current_P = P1
                current_depth = d
                # si on a trouvé un point, on arrête de chercher parmi les candidats
                found = True
                # si on descendait et qu'on a trouvé, on continue en descendant
                # sinon on remonte naturellement à la prochaine itération

    return path




# ============================================================
# VISUALISATION TOUS LES CHEMINS + CHEMIN LE PLUS LONG
# ============================================================

def plot_all_paths(curves, all_paths):
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='datalim')

    # tracer les courbes de niveau
    for d, c in curves.items():
        x, y = c.xy
        plt.plot(x, y, label=f"z={d}", alpha=0.6)

    # tracer tous les chemins en transparent
    for path in all_paths:
        xs = [p[0][0] for p in path]
        ys = [p[0][1] for p in path]
        plt.plot(xs, ys, 'gray', alpha=0.3)

    # tracer le chemin le plus long en rouge
    longest_path = max(all_paths, key=lambda p: len(p))
    xs = [p[0][0] for p in longest_path]
    ys = [p[0][1] for p in longest_path]
    plt.plot(xs, ys, 'red', lw=2, label="Chemin le plus long")

    plt.scatter(xs[0], ys[0], c="blue", s=80, label="Départ")
    plt.legend()
    plt.title("Chaîne de projections – Lake Aydat")
    plt.show()

# ============================================================
# EXECUTION SUR POINTS ÉCHANTILLONNÉS
# ============================================================

points_csv = "../donnee/curves/Lake_Aydat_points.csv"
lines_csv  = "../donnee/curves/Lake_Aydat_lines.csv"

points, segments, depths = read_curves(points_csv, lines_csv)
curves_by_depth = build_curves(points, segments, depths)
print(curves_by_depth[-16])

# --- profondeur de départ ---
depth0 = -4.0
curve0 = curves_by_depth[depth0]
num_points = len(curve0.coords)
N = 5  # prendre 1 point sur N pour accélérer

angle_max = np.deg2rad(10)

# --- calcul des projections pour points échantillonnés ---
all_paths = []
for idx in range(0, num_points, N):
    t = idx / (num_points - 1)
    P0 = np.array(curve0.interpolate(t, normalized=True).coords[0])
    path = find_projection_iterative(P0, depth0, curves_by_depth, angle_max, max_steps=15)
    all_paths.append(path)

# --- visualiser tous les chemins avec le plus long en rouge ---
plot_all_paths(curves_by_depth, all_paths)
