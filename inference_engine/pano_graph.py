import gtsam
import numpy as np

class PanoPoseGraph:
    def __init__(self):
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_estimates = gtsam.Values()
        self.optimized_values = gtsam.Values()
        
        self.odom_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.05, 0.05, 0.1, 0.1, 0.1]))
        self.prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-6] * 6))
        self.nodes = set()

    def _matrix_to_pose3(self, T: np.ndarray) -> gtsam.Pose3:
        R = gtsam.Rot3(T[:3, :3])
        t = gtsam.Point3(T[:3, 3])
        return gtsam.Pose3(R, t)

    def add_prior(self, node_id: int, pose_mat: np.ndarray):
        pose3 = self._matrix_to_pose3(pose_mat)
        self.graph.add(gtsam.PriorFactorPose3(node_id, pose3, self.prior_noise))
        self.initial_estimates.insert(node_id, pose3)
        self.nodes.add(node_id)

    def add_odometry(self, from_id: int, to_id: int, relative_mat: np.ndarray, initial_estimate_mat: np.ndarray):
        rel_pose3 = self._matrix_to_pose3(relative_mat)
        self.graph.add(gtsam.BetweenFactorPose3(from_id, to_id, rel_pose3, self.odom_noise))
        if to_id not in self.nodes:
            est_pose3 = self._matrix_to_pose3(initial_estimate_mat)
            self.initial_estimates.insert(to_id, est_pose3)
            self.nodes.add(to_id)

    def add_loop_closure(self, from_id: int, to_id: int, relative_mat: np.ndarray, noise_multiplier: float = 2.0):
        rel_pose3 = self._matrix_to_pose3(relative_mat)
        
        # [FIX]: Relax rotation noise massively (0.5 rad = ~30 deg) to trust odometry orientation, 
        # but keep translation noise tight (0.1m) to close the physical loop.
        lc_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([0.5, 0.5, 0.5, 0.1, 0.1, 0.1]) * noise_multiplier
        )
        self.graph.add(gtsam.BetweenFactorPose3(from_id, to_id, rel_pose3, lc_noise))

    def optimize(self):
        params = gtsam.LevenbergMarquardtParams()
        optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initial_estimates, params)
        self.optimized_values = optimizer.optimize()
        self.initial_estimates = self.optimized_values

    def get_optimized_pose(self, node_id: int) -> np.ndarray:
        if self.optimized_values.exists(node_id):
            return self.optimized_values.atPose3(node_id).matrix()
        return self.initial_estimates.atPose3(node_id).matrix()