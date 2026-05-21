import os
import tkinter as tk
from tkinter import filedialog
import numpy as np
import open3d as o3d

def create_camera_frustum(pose, size=0.2):
    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    camera_frame.transform(pose)
    return camera_frame

def load_npz_node(file_path):
    print(f"📦 Loading local node: {os.path.basename(file_path)}")
    data = np.load(file_path)
    
    # Flatten the points and colors from the arrays
    points = data['points'].reshape(-1, 3)
    colors = data['colors'].reshape(-1, 3) / 255.0
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    geometries = [{"name": "Point Cloud", "geometry": pcd}]
    
    # Auto-detect matching ground truth pose file
    dir_name = os.path.dirname(file_path)
    base_name = os.path.basename(file_path)
    
    if base_name.startswith("node_"):
        frame_idx = base_name.split("_")[1].split(".")[0]
        pose_path = os.path.join(dir_name, f"pose_{frame_idx}.npy")
        
        if os.path.exists(pose_path):
            print(f"📍 Found matching pose file: pose_{frame_idx}.npy")
            pose = np.load(pose_path)
            geometries.append({"name": f"Camera_{frame_idx}", "geometry": create_camera_frustum(pose, size=0.5)})
        else:
            print("⚠️ No matching pose file found in directory. Displaying point cloud only.")
            
    return geometries

def load_global_map(file_path):
    print(f"🌍 Loading global map: {os.path.basename(file_path)}")
    pcd = o3d.io.read_point_cloud(file_path)
    geometries = [{"name": "Global Map", "geometry": pcd}]
    
    dir_name = os.path.dirname(file_path)
    # Check for both possible names depending on which step you downloaded from
    traj_path_1 = os.path.join(dir_name, "trajectory.npy")
    traj_path_2 = os.path.join(dir_name, "FINAL_Trajectory.npy")
    
    traj_path = traj_path_1 if os.path.exists(traj_path_1) else traj_path_2
    
    if os.path.exists(traj_path):
        print("📍 Found matching trajectory! Loading camera path...")
        trajectory = np.load(traj_path)
        for i, pose in enumerate(trajectory):
            geometries.append({"name": f"Node_{i}", "geometry": create_camera_frustum(pose, size=0.3)})
    else:
        print("⚠️ No trajectory file found. Displaying map only.")
        
    return geometries

def main():
    root = tk.Tk()
    root.withdraw() 
    root.attributes('-topmost', True)

    print("📂 Please select a Point Cloud file to open...")
    
    current_directory = os.getcwd() 
    
    file_path = filedialog.askopenfilename(
        initialdir=current_directory,
        title="Select PanoVGGT Point Cloud",
        filetypes=[
            ("All Supported Files", "*.npz *.ply"),
            ("Single Node (.npz)", "*.npz"),
            ("Global Map (.ply)", "*.ply")
        ]
    )
    
    if not file_path:
        print("❌ No file selected. Exiting.")
        return

    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.npz':
        # Now calls the updated single-node loader
        geometries = load_npz_node(file_path)
    elif ext == '.ply':
        geometries = load_global_map(file_path)
    else:
        print(f"❌ Unsupported file type: {ext}")
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