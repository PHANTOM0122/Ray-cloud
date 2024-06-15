import numpy as np

def test_pt_idx(pts,ind_to_id,pts3d):
    ids = np.array([id for id in ind_to_id.values()])
    id_pts = np.array([pts3d[id].xyz for id in ids])
    print(np.equal(pts,id_pts).all())
    
def recontest_pt_idx(ptss,ind_to_ids,pts3D):
    print("Point ID-INDEX Consistency Test Module Initiated")
    for i in range(len(ptss)):
        print(f"Check point set [{i}] from same id")
        test_pt_idx(ptss[i],ind_to_ids[i],pts3D)
    print()
    
def compare_LPtest_PPLbase(ptss,lines):
    pts, pts2 = ptss
    subt = np.subtract(pts2,pts)
    subt /= np.linalg.norm(subt,axis=1,keepdims=True)+1e-7
    print("Check line pts match")
    print(np.isclose(subt,lines,rtol=1e-4).all())
    print(f"# of Point A : {len(pts)}")
    print(f"# of Point B : {len(pts2)}")
    print()