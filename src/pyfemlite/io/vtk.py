from __future__ import annotations
import numpy as np

def write_vtk_unstructured_tri(filename: str, X: np.ndarray, T: np.ndarray, point_data: dict[str, np.ndarray]):
    nnode = X.shape[0]
    nelem = T.shape[0]

    with open(filename, "w", encoding="utf-8") as f:
        f.write("# vtk DataFile Version 2.0\n")
        f.write("pyfemlite output\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")
        f.write(f"POINTS {nnode} float\n")
        for i in range(nnode):
            x, y = X[i]
            f.write(f"{x} {y} 0.0\n")

        f.write(f"CELLS {nelem} {nelem*4}\n")
        for e in range(nelem):
            n1, n2, n3 = T[e]
            f.write(f"3 {n1} {n2} {n3}\n")

        f.write(f"CELL_TYPES {nelem}\n")
        for _ in range(nelem):
            f.write("5\n")  # VTK_TRIANGLE

        if point_data:
            f.write(f"POINT_DATA {nnode}\n")
            for name, data in point_data.items():
                data = np.asarray(data)
                if data.ndim == 1:
                    f.write(f"SCALARS {name} float 1\nLOOKUP_TABLE default\n")
                    for v in data:
                        f.write(f"{float(v)}\n")
                elif data.ndim == 2 and data.shape[1] in (2, 3):
                    f.write(f"VECTORS {name} float\n")
                    for row in data:
                        if data.shape[1] == 2:
                            f.write(f"{float(row[0])} {float(row[1])} 0.0\n")
                        else:
                            f.write(f"{float(row[0])} {float(row[1])} {float(row[2])}\n")
                else:
                    raise ValueError(f"Unsupported point_data shape for {name}: {data.shape}")
