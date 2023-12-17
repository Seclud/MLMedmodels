import open3d as o3d
import os


def process_file(filename, output_directory):
    # Load the .obj file
    mesh = o3d.io.read_triangle_mesh(filename)

    # Convert the mesh to a point cloud
    point_cloud = mesh.sample_points_poisson_disk(number_of_points=3000)

    # Save the point cloud to a file
    base_filename = os.path.splitext(os.path.basename(filename))[0] + ".ply"
    output_filename = os.path.join(output_directory, base_filename)
    o3d.io.write_point_cloud(output_filename, point_cloud)

    # Visualize the point cloud
    # o3d.visualization.draw_geometries([point_cloud])
    print(f'Processed {counter} models out of {len(os.listdir("models"))}')


def main():
    global counter
    counter = 1
    input_directory = "models"
    output_directory = "Point clouds"
    for filename in os.listdir(input_directory):
        if filename.endswith(".obj"):
            process_file(os.path.join(input_directory, filename), output_directory)
            counter += 1


if __name__ == '__main__':
    main()