
from open3d import *

ply_fn = '/home/z/Research/suncg_v1/room/28297783bce682aac7fb35a1f35f68fa/parts/Wall#0_1_1.ply'
mesh = read_triangle_mesh(ply_fn)
vertices = np.asarray(mesh.vertices)
centroid = np.mean(vertices, 0)
mesh_frame = create_mesh_coordinate_frame(size = 0.6, origin = centroid)
draw_geometries([mesh, mesh_frame])
