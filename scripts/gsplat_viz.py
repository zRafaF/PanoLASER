import os
import time
import tkinter as tk
from tkinter import filedialog
import numpy as np
from plyfile import PlyData
import viser

def quat_to_rotmat(q):
    """Converts a batch of Quaternions (W, X, Y, Z) to 3x3 Rotation Matrices"""
    # Normalize quaternions first to prevent distortion
    norms = np.linalg.norm(q, axis=1, keepdims=True)
    q = q / (norms + 1e-8)
    
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.zeros((len(q), 3, 3), dtype=np.float32)
    
    R[:, 0, 0] = 1.0 - 2.0 * (y**2 + z**2)
    R[:, 0, 1] = 2.0 * (x*y - w*z)
    R[:, 0, 2] = 2.0 * (x*z + w*y)
    
    R[:, 1, 0] = 2.0 * (x*y + w*z)
    R[:, 1, 1] = 1.0 - 2.0 * (x**2 + z**2)
    R[:, 1, 2] = 2.0 * (y*z - w*x)
    
    R[:, 2, 0] = 2.0 * (x*z - w*y)
    R[:, 2, 1] = 2.0 * (y*z + w*x)
    R[:, 2, 2] = 1.0 - 2.0 * (x**2 + y**2)
    return R

def load_3dgs_ply(file_path):
    print(f"📦 Decoding 3DGS Binary PLY: {os.path.basename(file_path)}")
    t0 = time.time()
    
    plydata = PlyData.read(file_path)
    v = plydata['vertex']
    
    # 1. Extract Centers (Means)
    centers = np.vstack((v['x'], v['y'], v['z'])).T.astype(np.float32)
    
    # 2. Extract Colors from Spherical Harmonics (SH0)
    # The math inverse of: f_dc = (rgb - 0.5) / 0.28209
    f_dc = np.vstack((v['f_dc_0'], v['f_dc_1'], v['f_dc_2'])).T
    rgbs = (f_dc * 0.28209479177387814) + 0.5
    rgbs = np.clip(rgbs, 0.0, 1.0).astype(np.float32)
    
    # 3. Extract Opacities (Stored in Logit/Inverse-Sigmoid space)
    raw_opacities = v['opacity']
    opacities = 1.0 / (1.0 + np.exp(-raw_opacities))
    opacities = opacities.astype(np.float32).reshape(-1, 1)
    
    # 4. Extract Scales (Stored in Log space)
    raw_scales = np.vstack((v['scale_0'], v['scale_1'], v['scale_2'])).T
    scales = np.exp(raw_scales).astype(np.float32)
    
    # 5. Extract Quaternions and calculate Covariance Matrices
    quats = np.vstack((v['rot_0'], v['rot_1'], v['rot_2'], v['rot_3'])).T
    Rs = quat_to_rotmat(quats)
    
    # Sigma = R * S^2 * R^T 
    covariances = np.einsum(
        "nij,njk,nlk->nil", 
        Rs, 
        np.eye(3, dtype=np.float32)[None, :, :] * (scales[:, None, :] ** 2), 
        Rs
    )
    
    print(f"✅ Extracted {len(centers)} True Gaussians in {time.time()-t0:.2f}s")
    
    # FIX: Force all arrays to be C-contiguous in memory before handing off to Viser/WebGL
    return (
        np.ascontiguousarray(centers), 
        np.ascontiguousarray(rgbs), 
        np.ascontiguousarray(opacities), 
        np.ascontiguousarray(covariances)
    )

def main():
    # Setup Tkinter File Dialog
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    print("📂 Please select a 3D Gaussian Splat (.ply) file...")
    file_path = filedialog.askopenfilename(
        title="Select 3D Gaussian Splat PLY",
        filetypes=[("PLY Files", "*.ply")]
    )
    
    if not file_path:
        print("❌ No file selected. Exiting.")
        return

    try:
        centers, rgbs, opacities, covariances = load_3dgs_ply(file_path)
    except Exception as e:
        print(f"❌ Failed to parse PLY. Are you sure it's a 3DGS file? Error: {e}")
        return

    print("🚀 Launching Viser WebGL Server...")
    # Viser launches a lightweight local server to host the rasterizer
    server = viser.ViserServer(port=8080)
    
    # Inject the Gaussians into the Viser Scene
    server.scene.add_gaussian_splats(
        "/splats",
        centers=centers,
        rgbs=rgbs,
        opacities=opacities,
        covariances=covariances,
    )
    
    print("\n" + "="*55)
    print("🌟 3DGS RASTERIZER IS LIVE!")
    print("👉 Open your browser to: http://localhost:8080")
    print("="*55 + "\n")
    
    # Keep the main thread alive while the WebGL server runs in the background
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nShutting down viewer.")

if __name__ == "__main__":
    main()