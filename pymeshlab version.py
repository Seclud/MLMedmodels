import pymeshlab as ml

# Load the mesh
ms = ml.MeshSet()
ms.load_new_mesh('models/008738.obj')

# Sample the surface of the mesh
ms.apply_filter('sample_surface_poisson_disk', samplenum=2500)

# Get the sampled point cloud
point_cloud = ms.current_mesh()

# Save the point cloud to a file
ms.save_current_mesh('output.ply')