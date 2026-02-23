import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import nearest_points, linemerge, unary_union
from scipy.spatial.distance import cdist
from collections import defaultdict
import pandas as pd
import csv


# ------------------------
# LECTURE DES CSV
# ------------------------
def read_curves(points_csv, lines_csv):
    pts = []
    with open(points_csv, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            pts.append((float(row[0]), float(row[1]), float(row[2])))

    segs, depths = [], []
    with open(lines_csv, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            segs.append((int(row[0]), int(row[1])))
            depths.append(float(row[2]))

    if len(pts) == 0 or len(segs) == 0:
        raise ValueError("Les fichiers de courbes sont vides")

    return np.array(pts), np.array(segs), np.array(depths)



# ------------------------
# CONSTRUCTION DES COURBES
# ------------------------
def curves_to_closed_loops(points, lines, depths):
    segments_by_depth = defaultdict(list)

    for (i, j), d in zip(lines, depths):
        seg = LineString([(points[i, 0], points[i, 1]),
                          (points[j, 0], points[j, 1])])
        segments_by_depth[d].append(seg)

    closed_loops = {}
    for depth, segments in segments_by_depth.items():
        merged = linemerge(unary_union(segments))
        loops = []

        if merged.geom_type == "LineString":
            geoms = [merged]
        else:
            geoms = list(merged.geoms)

        for g in geoms:
            if g.is_ring:
                loops.append(g)

        closed_loops[depth] = loops

    return closed_loops


# ------------------------
# DISCRETISATION DES COURBES
# ------------------------
def discretize_curves(curves_by_depth, n_points_per_curve=50):
    points = []
    depths = []
    for depth, curves in curves_by_depth.items():
        for curve in curves:
            coords = list(curve.coords)
            n_coords = len(coords)
            idxs = np.linspace(0, n_coords-1, n_points_per_curve, dtype=int)
            for i in idxs:
                x, y = coords[i]
                points.append([x, y, depth])
                depths.append(depth)
    return np.array(points), np.array(depths)

# ------------------------
# CALCUL DES GRADIENTS 2D
# ------------------------
def compute_gradients(points, curves_by_depth):
    """
    points: Nx3 array (x, y, z)
    Retourne Nx2 array dz/dx, dz/dy
    """
    gradients = []
    depths = sorted(curves_by_depth.keys())
    for x, y, z in points:
        # chercher la courbe la plus proche à profondeur immédiatement supérieure
        next_depths = [d for d in depths if d > z]
        if not next_depths:
            gradients.append([0.0, 0.0])
            continue
        z_next = next_depths[0]
        next_curves = curves_by_depth[z_next]

        P = Point(x, y)
        min_dist = float("inf")
        nearest_Q = None
        for curve in next_curves:
            Q_candidate = nearest_points(P, curve)[1]
            dist_candidate = P.distance(Q_candidate)
            if dist_candidate < min_dist:
                min_dist = dist_candidate
                nearest_Q = Q_candidate
        if nearest_Q is None:
            gradients.append([0.0, 0.0])
            continue

        dx = nearest_Q.x - x
        dy = nearest_Q.y - y
        dz = z_next - z

        dist2d = np.sqrt(dx**2 + dy**2)
        if dist2d < 1e-6:
            dist2d = 1e-6

        grad_vec = np.array([dz * dx / dist2d**2, dz * dy / dist2d**2])
        gradients.append(grad_vec)
    return np.array(gradients)

# ------------------------
# HERMITE-RBF GENERAL
# ------------------------
def hermite_rbf_surface_full(points, gradients, curves_by_depth, grid_resolution=100):
    N = points.shape[0]
    xy = points[:, :2]
    z = points[:, 2]

    # RBF thin-plate spline
    def phi(r):
        with np.errstate(divide='ignore', invalid='ignore'):
            r2 = r**2
            result = r2 * np.log(r + 1e-12)
            result[np.isnan(result)] = 0.0
        return result

    D = cdist(xy, xy)
    K = phi(D)

    dx = xy[:,0][:,None] - xy[:,0][None,:]
    dy = xy[:,1][:,None] - xy[:,1][None,:]
    dK_dx = 2*dx*np.log(D + 1e-12) + dx
    dK_dy = 2*dy*np.log(D + 1e-12) + dy

    # least-squares pour stabilité
    A = np.block([
        [K, dK_dx, dK_dy],
        [dK_dx.T, np.zeros((N,N)), np.zeros((N,N))],
        [dK_dy.T, np.zeros((N,N)), np.zeros((N,N))]
    ])
    b = np.concatenate([z, gradients[:,0], gradients[:,1]])
    coeffs = np.linalg.lstsq(A, b, rcond=1e-6)[0]
    c = coeffs[:N]
    cx = coeffs[N:2*N]
    cy = coeffs[2*N:]

    # Masque polygonal : union de toutes les courbes
    polygons = []
    for curves in curves_by_depth.values():
        for loop in curves:
            if loop.is_ring:
                polygons.append(Polygon(loop))
    domain_polygon = unary_union(polygons)
    minx, miny, maxx, maxy = domain_polygon.bounds

    # grille
    gx = np.linspace(minx, maxx, grid_resolution)
    gy = np.linspace(miny, maxy, grid_resolution)
    GX, GY = np.meshgrid(gx, gy)
    grid_points = np.column_stack([GX.ravel(), GY.ravel()])

    # évaluation RBF Hermite
    R = phi(cdist(grid_points, xy))
    dR_dx = 2*(grid_points[:,0][:,None] - xy[:,0][None,:]) * np.log(cdist(grid_points, xy) + 1e-12) + (grid_points[:,0][:,None] - xy[:,0][None,:])
    dR_dy = 2*(grid_points[:,1][:,None] - xy[:,1][None,:]) * np.log(cdist(grid_points, xy) + 1e-12) + (grid_points[:,1][:,None] - xy[:,1][None,:])

    GZ = R.dot(c) + dR_dx.dot(cx) + dR_dy.dot(cy)

    # filtrage intérieur
    result_points = []
    for (x, y), z_val in zip(grid_points, GZ):
        if domain_polygon.covers(Point(x, y)):
            result_points.append([x, y, z_val])

    return np.array(result_points)

# ------------------------
# EXPORT CSV
# ------------------------
def save_point_cloud(points, output_csv):
    import pandas as pd
    pd.DataFrame(points, columns=["x","y","z"]).to_csv(output_csv, index=False)

# ------------------------
# PIPELINE COMPLET
# ------------------------
def reconstruct_lake_full(points_csv, lines_csv, output_csv, n_points_per_curve=50, grid_resolution=100):
    import numpy as np
    from collections import defaultdict

    # lecture
    points, segments, depths = read_curves(points_csv, lines_csv)
    curves_by_depth = curves_to_closed_loops(points, segments, depths)

    # discrétisation toutes courbes
    surface_points, surface_depths = discretize_curves(curves_by_depth, n_points_per_curve)
    gradients = compute_gradients(surface_points, curves_by_depth)

    # Hermite-RBF global
    pts = hermite_rbf_surface_full(surface_points, gradients, curves_by_depth, grid_resolution=grid_resolution)

    # fusion avec points initiaux
    all_points = np.vstack([points, pts])
    save_point_cloud(all_points, output_csv)
    print(f"Nuage de points interpolé sur toute la figure sauvegardé dans {output_csv}")


# ------------------------
# EXECUTION
# ------------------------
if __name__ == "__main__":
    reconstruct_lake_full(
        points_csv="../donnee/curves/Lake_Aydat_points.csv",
        lines_csv="../donnee/curves/Lake_Aydat_lines.csv",
        output_csv="../donnee/point_cloud/Lake_Aydat_better.csv",
    )