#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pymeshlab
import sys

def edge_collapse(input_mesh, output_mesh, target_faces=5000):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(input_mesh)

    print("Nombre de faces initial :", ms.current_mesh().face_number())

    # Simplification par Quadric Edge Collapse
    ms.meshing_decimation_quadric_edge_collapse(
        targetfacenum=target_faces,
        preservenormal=True,
        preservetopology=True,
        preserveboundary=True,
        optimalplacement=True
    )

    print("Nombre de faces après simplification :", ms.current_mesh().face_number())

    ms.save_current_mesh(output_mesh)
    print("Maillage sauvegardé :", output_mesh)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage : python edge_collapse.py input.obj output.obj")
        sys.exit(1)

    edge_collapse(sys.argv[1], sys.argv[2])
