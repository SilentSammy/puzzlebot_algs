import threading
import numpy as np
import cv2
import math

def vecs_to_matrix(rvec, tvec):
    """Convert rvec, tvec to a 4x4 transformation matrix."""
    rvec = np.asarray(rvec, dtype=np.float32)
    tvec = np.asarray(tvec, dtype=np.float32)
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    return T

def matrix_to_vecs(T):
    """Convert a 4x4 transformation matrix to rvec, tvec."""
    R = T[:3, :3]
    tvec = T[:3, 3]
    rvec, _ = cv2.Rodrigues(R)
    return rvec.flatten(), tvec.flatten()

class PnpResult:
    def __init__(self, obj_pts, img_pts, tvec, rvec):
        """
        obj_pts: array of shape (N, 1, 3) or (N, 3) containing 3D object‐space coordinates
                 (X, Y, Z) of detected Charuco corners (Z is usually 0).
        img_pts: array of shape (N, 1, 2) or (N, 2) containing 2D image‐space coordinates (u, v).
        tvec, rvec: the usual solvePnP outputs (not used in project_point).
        """
        # Convert obj_pts to shape (N, 2) by flattening and taking X, Y only
        obj = np.asarray(obj_pts, dtype=np.float32)
        if obj.ndim == 3 and obj.shape[1] == 1 and obj.shape[2] == 3:
            obj = obj.reshape(-1, 3)
        elif obj.ndim == 2 and obj.shape[1] == 3:
            pass
        else:
            raise ValueError(f"Unexpected obj_pts shape {obj.shape}, expected (N,1,3) or (N,3)")

        # Only keep X, Y columns
        self.obj_pts = obj[:, :2].copy()  # shape (N, 2)

        # Convert img_pts to shape (N, 2)
        img = np.asarray(img_pts, dtype=np.float32)
        if img.ndim == 3 and img.shape[1] == 1 and img.shape[2] == 2:
            img = img.reshape(-1, 2)
        elif img.ndim == 2 and img.shape[1] == 2:
            pass
        else:
            raise ValueError(f"Unexpected img_pts shape {img.shape}, expected (N,1,2) or (N,2)")

        self.img_pts = img.copy()  # shape (N, 2)

        self.tvec = tvec
        self.rvec = rvec

    def get_ref_T(self):
        """Get the 4x4 transformation matrix from of the reference relative to the camera.
        
        Returns:
            4x4 numpy array representing the board pose
        """
        return vecs_to_matrix(self.rvec, self.tvec)

    def get_quad_corners(self):
        """
        Selects four corners from obj_pts/img_pts that correspond to the board's
        outer quadrilateral. Returns (quad_obj_pts, quad_img_pts), each shape (4, 2).
        """
        N = self.obj_pts.shape[0]
        if N < 4:
            raise ValueError("Need at least 4 points to form a quadrilateral")

        xs = self.obj_pts[:, 0]
        ys = self.obj_pts[:, 1]
        min_x, max_x = float(xs.min()), float(xs.max())
        min_y, max_y = float(ys.min()), float(ys.max())

        # Define the four ideal corner positions in object space:
        targets = [
            (min_x, min_y),  # top-left
            (max_x, min_y),  # top-right
            (max_x, max_y),  # bottom-right
            (min_x, max_y),  # bottom-left
        ]

        quad_obj = []
        quad_img = []
        used_indices = set()

        for tx, ty in targets:
            diffs = self.obj_pts - np.array([tx, ty], dtype=np.float32)
            d2 = np.sum(diffs**2, axis=1)  # squared distance to each obj_pt
            idx = int(np.argmin(d2))

            if idx in used_indices:
                # If already used, pick the next closest unused
                sorted_idxs = np.argsort(d2)
                for candidate in sorted_idxs:
                    if candidate not in used_indices:
                        idx = int(candidate)
                        break

            used_indices.add(idx)
            quad_obj.append(self.obj_pts[idx])
            quad_img.append(self.img_pts[idx])

        quad_obj = np.array(quad_obj, dtype=np.float32)  # shape (4,2)
        quad_img = np.array(quad_img, dtype=np.float32)  # shape (4,2)
        return quad_obj, quad_img

    def project_point(self, point, z=0.0):
        quad_obj, quad_img = self.get_quad_corners()
        H = cv2.getPerspectiveTransform(quad_img, quad_obj)
        pts = np.array([[[point[0], point[1]]]], dtype=np.float32)  # shape (1,1,2)
        projected = cv2.perspectiveTransform(pts, H)  # shape (1,1,2)
        X = float(projected[0, 0, 0])
        Y = float(projected[0, 0, 1])
        Z = z

        # Get camera position in board coordinates
        board_T = self.get_ref_T()
        # board_T transforms from board to camera, so invert to get camera pose in board frame
        cam_T_in_board = np.linalg.inv(board_T)
        cam_pos = cam_T_in_board[:3, 3]  # Camera position in board coordinates
        
        # Calculate angle between camera and point
        # Vector from camera to point (not point to camera)
        delta_x = X - cam_pos[0]
        delta_y = Y - cam_pos[1]
        delta_z = Z - cam_pos[2]
        
        # Angle in X axis: angle between projection onto YZ plane
        # atan2(delta_x, delta_z) gives angle from Z axis toward X
        angle_x = math.atan2(delta_x, delta_z)
        
        # Angle in Y axis: angle between projection onto XZ plane
        angle_y = math.atan2(delta_y, delta_z)
        
        # Apply parallax correction based on object height and viewing angles
        # The homography projects to Z=0, but object is at height Z
        # Offset is Z * tan(angle) in each axis
        offset_x = Z * math.tan(angle_x)
        offset_y = Z * math.tan(angle_y)
        
        # Correct the position
        X_corrected = X - offset_x
        Y_corrected = Y - offset_y

        return (X_corrected, Y_corrected)

class PoseEstimator:
    def __init__(self, reference, K, D=None):
        """Initialize PoseEstimator.
        
        Args:
            reference: Reference target detector (ArucoDetector or GridboardDetector)
            K: Camera intrinsic matrix
            D: Distortion coefficients (default: zeros)
            rotate_180: Whether to rotate input frame 180° before processing (default: True)
        """
        self.reference = reference
        self.K = K
        self.D = D if D is not None else np.zeros(5)

    def get_pose(self, frame, drawing_frame=None):
        """Estimate camera pose relative to the reference target.
        
        Args:
            frame: Input frame
            drawing_frame: Optional frame to draw pose info on
            
        Returns:
            (pose_T, pnp_result) or None if detection fails
        """
        detection = self.reference.detect(frame, drawing_frame=drawing_frame)
        if detection is None:
            return None

        img_pts = detection.img_pts.copy()
        h, w = frame.shape[:2]
        
        # ===== IMAGE POINT ROTATION (before solvePnP) =====
        # Uncomment ONE of these to test different rotations:
        
        # No rotation
        # pass
        
        # 90° clockwise
        # img_pts_new = img_pts.copy()
        # img_pts_new[:, 0] = h - img_pts[:, 1]
        # img_pts_new[:, 1] = img_pts[:, 0]
        # img_pts = img_pts_new
        
        # 90° counter-clockwise
        # img_pts_new = img_pts.copy()
        # img_pts_new[:, 0] = img_pts[:, 1]
        # img_pts_new[:, 1] = w - img_pts[:, 0]
        # img_pts = img_pts_new
        
        # 180° rotation
        img_pts[:, 0] = w - img_pts[:, 0]
        img_pts[:, 1] = h - img_pts[:, 1]

        res = solve_pnp(detection.obj_pts, img_pts, self.K, self.D)
        if res is None:
            return None

        pose_T = vecs_to_matrix(res.rvec, res.tvec)

        # Apply pose rotation if detector requires it (e.g., GridboardDetector)
        if self.reference.needs_rot:
            R_x_180 = np.array([
                [1,  0,  0,  0],
                [0, -1,  0,  0],
                [0,  0, -1,  0],
                [0,  0,  0,  1]
            ], dtype=np.float64)
            pose_T = pose_T @ R_x_180

        if drawing_frame is not None:
            rvec, tvec = matrix_to_vecs(pose_T)
            rvec_string = ', '.join([str(round(math.degrees(x), 3)) for x in rvec])
            tvec_string = ', '.join([str(round(float(x), 3)) for x in tvec])
            cv2.putText(drawing_frame, f"R: {rvec_string}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
            cv2.putText(drawing_frame, f"T: {tvec_string}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

        return pose_T, res, detection
    
    def project_point(self, pnp_result, image_point, frame_shape, z=0.0):
        """Project image point to reference coordinates, handling rotation if enabled.
        
        Args:
            pnp_result: PnpResult from get_pose
            image_point: (x, y) tuple in original image coordinates
            frame_shape: (height, width) or (height, width, channels) of the frame
            
        Returns:
            (X, Y) in reference coordinates
        """
        if self.rotate_180:
            h, w = frame_shape[:2]
            cx, cy = w / 2, h / 2
            x, y = image_point
            rotated_point = (2 * cx - x, 2 * cy - y)
            ref_x, ref_y = pnp_result.project_point(rotated_point, z=z)
            # Invert Y to match reference coordinate convention (Y up vs image Y down)
            return (ref_x, -ref_y)
        else:
            return pnp_result.project_point(image_point, z=z)

def solve_pnp(obj_pts, img_pts, K, D, flags=cv2.SOLVEPNP_ITERATIVE):
    success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, D, flags=flags)
    if not success:
        return None
    return PnpResult(obj_pts=obj_pts, img_pts=img_pts, rvec=rvec.flatten(), tvec=tvec.flatten())

def get_cam_T(ref_T: np.ndarray) -> np.ndarray:
    # Invert to get camera-to-reference
    cam_T = np.linalg.inv(ref_T)

    return cam_T

class PosePlotter3D:
    """Real-time 3D visualization of camera pose relative to a reference target.
    
    Can display either:
    - Camera at origin, reference moving (camera_at_origin=True, default)
    - Reference at origin, camera moving (camera_at_origin=False)
    """
    
    def __init__(self, reference, axis_limit=1.0, update_interval=10, camera_at_origin=True):
        """Initialize 3D plotter.
        
        Args:
            reference: Reference target detector to get dimensions
            axis_limit: Axis limits in meters (default: 1.0m cube)
            update_interval: Update plot every N frames (default: 10)
            camera_at_origin: If True, camera at origin and reference moves.
                            If False, reference at origin and camera moves (default: True)
        """
        # Lazy import matplotlib only when plotter is created
        import matplotlib
        matplotlib.use('TkAgg')  # Use non-threaded backend
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        
        self.plt = plt
        self.Poly3DCollection = Poly3DCollection
        
        self.reference = reference
        self.axis_limit = axis_limit
        self.update_interval = update_interval
        self.camera_at_origin = camera_at_origin
        self.frame_count = 0
        
        # Get reference dimensions
        self.ref_width, self.ref_height = reference.get_board_dimensions()
        
        # Setup plot
        plt.ion()
        self.fig = plt.figure(figsize=(8, 6))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Initialize artists (will be updated)
        self.ref_poly = None
        self.ref_quivers = []
        self.camera_artists = []
        
        self._setup_plot()
        
        # Draw fixed reference frame based on mode
        if self.camera_at_origin:
            self._draw_camera_frame()
        else:
            self._draw_reference_frame()
        
        # Initialize moving object artists
        self.ref_poly = None
        self.ref_quivers = []
        
        # Show initially
        self.plt.show(block=False)
        self.plt.pause(0.001)
        
    def _setup_plot(self):
        """Configure 3D axes and labels."""
        self.ax.set_xlim([-self.axis_limit, self.axis_limit])
        self.ax.set_ylim([-self.axis_limit, self.axis_limit])
        self.ax.set_zlim([0, 2 * self.axis_limit])
        
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        self.ax.set_title('Camera Pose Estimation')
        
        # Set viewing angle
        self.ax.view_init(elev=20, azim=45)
        
    def _draw_camera_frame(self):
        """Draw camera coordinate frame at origin (only once)."""
        axis_length = 0.2
        
        # Camera coordinate axes at origin
        self.camera_artists.append(
            self.ax.quiver(0, 0, 0, axis_length, 0, 0, color='r', arrow_length_ratio=0.3, linewidth=2)
        )
        self.camera_artists.append(
            self.ax.quiver(0, 0, 0, 0, axis_length, 0, color='g', arrow_length_ratio=0.3, linewidth=2)
        )
        self.camera_artists.append(
            self.ax.quiver(0, 0, 0, 0, 0, axis_length, color='b', arrow_length_ratio=0.3, linewidth=2)
        )
    
    def _draw_reference_frame(self):
        """Draw reference coordinate frame and plane at world_T (only once)."""
        axis_length = 0.2
        world_T = self.reference.world_T
        origin = world_T[:3, 3]
        x_axis = world_T[:3, 0] * axis_length
        y_axis = world_T[:3, 1] * axis_length
        z_axis = world_T[:3, 2] * axis_length

        self.camera_artists.append(
            self.ax.quiver(*origin, *x_axis, color='r', arrow_length_ratio=0.3, linewidth=2)
        )
        self.camera_artists.append(
            self.ax.quiver(*origin, *y_axis, color='g', arrow_length_ratio=0.3, linewidth=2)
        )
        self.camera_artists.append(
            self.ax.quiver(*origin, *z_axis, color='b', arrow_length_ratio=0.3, linewidth=2)
        )

        # Draw reference plane at world_T
        corners_local = self._get_reference_corners()
        corners_world = self._transform_points(corners_local, world_T)
        verts = [corners_world]
        ref_poly = self.Poly3DCollection(verts, alpha=0.3, facecolor='gray', edgecolor='black', linewidth=2)
        self.ax.add_collection3d(ref_poly)
        self.camera_artists.append(ref_poly)
        
    def _get_reference_corners(self):
        """Get reference corners in reference's local frame.
        
        Returns:
            np.ndarray: 4x3 array of corner positions (centered at origin)
        """
        w, h = self.ref_width, self.ref_height
        
        # Corners centered at origin, in XY plane (Z=0)
        corners = np.array([
            [-w/2, -h/2, 0],  # Bottom-left
            [ w/2, -h/2, 0],  # Bottom-right
            [ w/2,  h/2, 0],  # Top-right
            [-w/2,  h/2, 0],  # Top-left
        ])
        
        return corners
    
    def _transform_points(self, points, T):
        """Transform points by homogeneous transformation matrix.
        
        Args:
            points: Nx3 array of points
            T: 4x4 transformation matrix
            
        Returns:
            Nx3 array of transformed points
        """
        # Convert to homogeneous coordinates
        points_h = np.hstack([points, np.ones((points.shape[0], 1))])
        
        # Apply transformation
        points_transformed = (T @ points_h.T).T
        
        # Convert back to 3D
        return points_transformed[:, :3]
    
    def update(self, pose_T):
        """Update visualization with new camera pose.
        
        Args:
            pose_T: 4x4 homogeneous transformation matrix (camera pose relative to reference)
        """
        # Only update every N frames to reduce lag
        self.frame_count += 1
        if self.frame_count % self.update_interval != 0:
            return
        
        # If reference is at origin, invert the transform to show camera moving
        if not self.camera_at_origin:
            pose_T = self.reference.world_T @ np.linalg.inv(pose_T)
        
        # Remove old visualization
        if self.ref_poly is not None:
            self.ref_poly.remove()
        for quiver in self.ref_quivers:
            quiver.remove()
        self.ref_quivers.clear()
        
        # Draw based on mode
        if self.camera_at_origin:
            # Draw reference plane and axes at transformed position
            self._draw_moving_reference(pose_T)
        else:
            # Draw camera axes only (no plane) at transformed position
            self._draw_moving_camera(pose_T)
        
        # Refresh display (non-blocking)
        self.fig.canvas.flush_events()
    
    def _draw_moving_reference(self, pose_T):
        """Draw reference plane and coordinate frame at given transform."""
        # Get reference corners and transform to camera frame
        corners_local = self._get_reference_corners()
        corners_camera = self._transform_points(corners_local, pose_T)
        
        # Draw reference as filled polygon
        verts = [corners_camera]
        self.ref_poly = self.Poly3DCollection(verts, alpha=0.5, facecolor='cyan', edgecolor='darkblue', linewidth=2)
        self.ax.add_collection3d(self.ref_poly)
        
        # Draw reference coordinate frame
        ref_origin = pose_T[:3, 3]
        axis_length = 0.15
        
        # Extract rotation axes from transformation matrix
        x_axis = pose_T[:3, 0] * axis_length
        y_axis = pose_T[:3, 1] * axis_length
        z_axis = pose_T[:3, 2] * axis_length
        
        # Draw axes
        self.ref_quivers.append(
            self.ax.quiver(ref_origin[0], ref_origin[1], ref_origin[2],
                          x_axis[0], x_axis[1], x_axis[2],
                          color='r', arrow_length_ratio=0.3, linewidth=1.5, alpha=0.7)
        )
        self.ref_quivers.append(
            self.ax.quiver(ref_origin[0], ref_origin[1], ref_origin[2],
                          y_axis[0], y_axis[1], y_axis[2],
                          color='g', arrow_length_ratio=0.3, linewidth=1.5, alpha=0.7)
        )
        self.ref_quivers.append(
            self.ax.quiver(ref_origin[0], ref_origin[1], ref_origin[2],
                          z_axis[0], z_axis[1], z_axis[2],
                          color='b', arrow_length_ratio=0.3, linewidth=1.5, alpha=0.7)
        )
    
    def _draw_moving_camera(self, cam_T):
        """Draw camera coordinate frame (axes only) at given transform."""
        camera_origin = cam_T[:3, 3]
        axis_length = 0.15
        
        # Extract rotation axes from transformation matrix
        x_axis = cam_T[:3, 0] * axis_length
        y_axis = cam_T[:3, 1] * axis_length
        z_axis = cam_T[:3, 2] * axis_length
        
        # Draw axes
        self.ref_quivers.append(
            self.ax.quiver(camera_origin[0], camera_origin[1], camera_origin[2],
                          x_axis[0], x_axis[1], x_axis[2],
                          color='r', arrow_length_ratio=0.3, linewidth=1.5, alpha=0.7)
        )
        self.ref_quivers.append(
            self.ax.quiver(camera_origin[0], camera_origin[1], camera_origin[2],
                          y_axis[0], y_axis[1], y_axis[2],
                          color='g', arrow_length_ratio=0.3, linewidth=1.5, alpha=0.7)
        )
        self.ref_quivers.append(
            self.ax.quiver(camera_origin[0], camera_origin[1], camera_origin[2],
                          z_axis[0], z_axis[1], z_axis[2],
                          color='b', arrow_length_ratio=0.3, linewidth=1.5, alpha=0.7)
        )
    
    def close(self):
        """Close the plot window."""
        self.plt.close(self.fig)

if __name__ == "__main__":
    import cv2
    from marker_det import global_detector, board_config_letter, ArucoDetector
    from video_source import CameraIntrinsics, CaptureSource, FileSource
    from sources import webcam, wide_angle_3

    # cap = webcam
    # cap = wide_angle_3
    cap = FileSource(r"recordings\recording_1777260746.mp4", loop=True)
    
    # reference = global_board_config
    aruco = ArucoDetector(
        dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
        marker_id=0,
        marker_size=0.1
    )
    # # world_T: pose of the ArUco marker in world space.
    # # Translation: x=0.5, y=0, z=0.05 (meters)
    # # Orientation: ZYX Euler angles — Alpha(Z)=0°, Beta(Y)=-90°, Gamma(X)=-90°
    # _a, _b, _g = np.radians([-90, -90, 0])
    # _Rz = np.array([[np.cos(_a), -np.sin(_a), 0], [np.sin(_a), np.cos(_a), 0], [0, 0, 1]])
    # _Ry = np.array([[np.cos(_b), 0, np.sin(_b)], [0, 1, 0], [-np.sin(_b), 0, np.cos(_b)]])
    # _Rx = np.array([[1, 0, 0], [0, np.cos(_g), -np.sin(_g)], [0, np.sin(_g), np.cos(_g)]])
    # aruco.world_T = np.eye(4)
    # aruco.world_T[:3, :3] = _Rz @ _Ry @ _Rx
    # aruco.world_T[:3, 3] = [0.5, 0, 0.05]
    reference = board_config_letter
    reference = aruco

    estimator = PoseEstimator(
        reference=reference,
        K=cap.get_intrinsics().K,
        D=cap.get_intrinsics().D
    )
    
    plotter = PosePlotter3D(
        reference,
        axis_limit=0.5,
        # camera_at_origin=True,
        camera_at_origin=False,
    )
    
    while True:
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
        # Get frame
        ret, frame = cap.read()
        if not ret:
            continue
        drawing_frame = frame.copy()

        # Estimate pose
        res = estimator.get_pose(frame, drawing_frame=drawing_frame)
    
        if res is not None:
            pose_T, _ = res
            plotter.update(pose_T)
    
        # Display
        cv2.imshow("Camera", drawing_frame)
