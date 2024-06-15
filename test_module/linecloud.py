import numpy as np

def line_integrity_test(line_to_pts, pts_to_line, line_3d, pts_3d_query):
    print("Line Integrity Test Module Initiated")
    for k, v in line_to_pts.items():
        p1, p2 = v
        _d = np.subtract(pts_3d_query[p1].xyz, pts_3d_query[p2].xyz)

        if not np.isclose(_d[0] / line_3d[k][0], _d[1] / line_3d[k][1]):
            print("Line1 : ", pts_to_line[p1])
            print("Line2 : ", pts_to_line[p2])
            print("Line3 : ", line_3d[k][1])
            if np.isclose(np.linalg.norm(_d), 0):
                continue
            else:
                raise Exception("Wrong Line Mapping")

        for i in range(3):
            if pts_to_line[p1][i] != pts_to_line[p2][i]:
                raise Exception("Wrong Line Mapping")
            if line_3d[k][i] != pts_to_line[p1][i]:
                raise Exception("Wrong Line Mapping")
    print("Line consistency : Fine\n")