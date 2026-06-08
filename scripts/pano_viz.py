import os
import tkinter as tk
from tkinter import filedialog
import numpy as np
import open3d as o3d

def load_npz_node(file_path):
    print(f"📦 Loading local node: {os.path.basename(file_path)}")
    data = np.load(file_path)
    
    # Flatten the points and colors from the arrays
    points = data['points'].reshape(-1, 3)
    colors = data['colors'].reshape(-1, 3) / 255.0
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return [{"name": "Point Cloud", "geometry": pcd}]

def load_global_map(file_path):
    print(f"🌍 Loading global map: {os.path.basename(file_path)}")
    pcd = o3d.io.read_point_cloud(file_path)
    
    return [{"name": "Global Map", "geometry": pcd}]

def load_glb_model(file_path):
    print(f"🧊 Loading 3D model: {os.path.basename(file_path)}")
    
    # Attempt 1: Native Open3D
    mesh = o3d.io.read_triangle_mesh(file_path)
    
    # If Open3D fails due to Assimp buffer issues, it returns a mesh with 0 vertices
    if not mesh.has_vertices():
        print("⚠️ Open3D failed to parse GLB buffers. Falling back to Trimesh...")
        try:
            import trimesh
        except ImportError:
            print("❌ 'trimesh' is required to load this GLB. Please run: pip install trimesh")
            return []
            
        # Attempt 2: Trimesh Fallback
        # force='mesh' attempts to load the scene directly as a single mesh object
        scene_or_mesh = trimesh.load(file_path, force='mesh')
        
        # If Trimesh returns a Scene with multiple objects, flatten it into one mesh
        if isinstance(scene_or_mesh, trimesh.Scene):
            geom = scene_or_mesh.dump(concatenate=True)
        else:
            geom = scene_or_mesh
            
        # Convert Trimesh data structures back into Open3D structures
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(geom.vertices)
        mesh.triangles = o3d.utility.Vector3iVector(geom.faces)
        
        # Pull vertex colors if the GLB has them
        if hasattr(geom.visual, 'vertex_colors') and geom.visual.vertex_colors is not None:
            # Trimesh stores colors as 0-255, Open3D needs 0.0-1.0
            colors = geom.visual.vertex_colors[:, :3] / 255.0
            mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
            
    # Compute vertex normals to ensure proper lighting in the viewer
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
        
    return [{"name": "GLB Model", "geometry": mesh}]

def main():
    root = tk.Tk()
    root.withdraw() 
    root.attributes('-topmost', True)

    print("📂 Please select a Point Cloud or 3D Model file to open...")
    
    current_directory = os.getcwd() 
    
    file_path = filedialog.askopenfilename(
        initialdir=current_directory,
        title="Select 3D File",
        filetypes=[
            ("All Supported Files", "*.npz *.ply *.glb"),
            ("Single Node (.npz)", "*.npz"),
            ("Global Map (.ply)", "*.ply"),
            ("3D Model (.glb)", "*.glb")
        ]
    )
    
    if not file_path:
        print("❌ No file selected. Exiting.")
        return

    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.npz':
        geometries = load_npz_node(file_path)
    elif ext == '.ply':
        geometries = load_global_map(file_path)
    elif ext == '.glb':
        geometries = load_glb_model(file_path)
    else:
        print(f"❌ Unsupported file type: {ext}")
        return

    # Check if geometries were successfully loaded before launching viewer
    if not geometries:
        print("❌ Failed to load any geometry. Exiting.")
        return

    print("🚀 Launching Modern 3D Viewer...")
    print("   -> Look for the UI panel on the right.")
    print("   -> Change 'Mouse control' from 'Arcball' to 'Fly'.")
    print("   -> Use WASD and your mouse to fly through the scene!")
    
    o3d.visualization.draw(
        geometries, 
        title="Equi-360 Visualizer", 
        bg_color=(0.05, 0.05, 0.05, 1.0),
        show_ui=True
    )

if __name__ == "__main__":
    main()