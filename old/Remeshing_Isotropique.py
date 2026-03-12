import pyvista as pv
import sys

def isotropic_remeshing(input_mesh, output_mesh, subdivision_level=1, reduction=0.3):
    mesh = pv.read(input_mesh)

    print("Nombre de cellules initial :", mesh.n_cells)

    mesh = mesh.clean()

    # Régularisation
    mesh = mesh.subdivide(subdivision_level, subfilter='loop')

    # Réduction contrôlée
    mesh = mesh.decimate(reduction)

    print("Nombre de cellules après remeshing :", mesh.n_cells)

    mesh.save(output_mesh)
    print("Maillage sauvegardé :", output_mesh)


if __name__ == "__main__":
    isotropic_remeshing(sys.argv[1], sys.argv[2])
