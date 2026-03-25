import numpy as np
from dataclasses import dataclass
from stereo_calibration import StereoCalibration
from common_lab_utils import PerspectiveCamera
import cv2
import pyvista as pv
from pylie import SE3


class Scene3D:
    """Visualises the lab in 3D"""

    _do_exit = False
    _current_camera_actors = ()

    def __init__(self, stereo_model: StereoCalibration):
        """Sets up the 3D viewer"""

        self._stereo_model = stereo_model
        self._plotter = pv.Plotter()
        self._point_cloud_actors = None

        # Left camera.
        left_camera_model = PerspectiveCamera(stereo_model.k_left, stereo_model.distortion_left, SE3())
        add_frustum(self._plotter, left_camera_model, stereo_model.img_size, 0.02)
        add_axis(self._plotter, left_camera_model.pose_world_camera, 0.02)

        # Right camera.
        right_camera_model = PerspectiveCamera(stereo_model.k_right, stereo_model.distortion_right, stereo_model.pose_left_right)
        add_frustum(self._plotter, right_camera_model, stereo_model.img_size, 0.02)
        add_axis(self._plotter, right_camera_model.pose_world_camera, 0.02)

        # Set camera.
        self._plotter.camera.position = (-0.017032, -0.160165, -0.518498)
        self._plotter.camera.up = (0.0189573, -0.978103, 0.207257)
        self._plotter.camera.focal_point = (-0.0355778, 0.0142925, 0.306513)

        # Show window.
        self._plotter.show(title="3D visualisation", interactive=True, interactive_update=True)

    def _update_current_camera_visualisation(self, pts_3d: np.ndarray):
        # Remove old visualisation.
        if self._point_cloud_actors is not None:
            self._plotter.remove_actor(self._point_cloud_actors, render=False)

        if len(pts_3d) > 0:
            # Render new visualisation.
            point_cloud = pv.PolyData(pts_3d)
            point_cloud['Depth'] = pts_3d[:, -1]

            self._point_cloud_actors = self._plotter.add_mesh(point_cloud, render_points_as_spheres=True)

    def update(self, pts_3d, time=10):
        """Updates the viewer with new point cloud"""

        self._update_current_camera_visualisation(pts_3d.T)
        self._plotter.update(time)


def add_axis(plotter, pose: SE3, scale=10.0):
    """Adds a 3D axis object to the pyvista plotter"""

    T = pose.to_matrix()

    point = pv.Sphere(radius=0.1 * scale).transform(T, inplace=False)
    x_arrow = pv.Arrow(direction=(1.0, 0.0, 0.0), scale=scale).transform(T, inplace=False)
    y_arrow = pv.Arrow(direction=(0.0, 1.0, 0.0), scale=scale).transform(T, inplace=False)
    z_arrow = pv.Arrow(direction=(0.0, 0.0, 1.0), scale=scale).transform(T, inplace=False)

    axis_actors = (
        plotter.add_mesh(point),
        plotter.add_mesh(x_arrow, color='red', render=False),
        plotter.add_mesh(y_arrow, color='green', render=False),
        plotter.add_mesh(z_arrow, color='blue', render=False)
    )
    return axis_actors


def add_frustum(plotter, camera_model, image_size, scale=0.1):
    """Adds a camera frustum to the pyvista plotter"""

    S = camera_model.pose_world_camera.to_matrix() @ np.diag([scale, scale, scale, 1.0])

    img_width, img_height = image_size

    point_bottom_left = np.squeeze(camera_model.pixel_to_normalised(np.array([img_width-1., img_height-1.])))
    point_bottom_right = np.squeeze(camera_model.pixel_to_normalised(np.array([0., img_height-1.])))
    point_top_left = np.squeeze(camera_model.pixel_to_normalised(np.array([0., 0.])))
    point_top_right = np.squeeze(camera_model.pixel_to_normalised(np.array([img_width-1., 0.])))

    point_focal = np.zeros([3])

    pyramid = pv.Pyramid(
        [point_bottom_left, point_bottom_right, point_top_left, point_top_right, point_focal]
    ).transform(S, inplace=False)

    frustum_actor = plotter.add_mesh(pyramid, show_edges=True, style='wireframe', render=False)

    return frustum_actor


@dataclass
class Colours:
    green = (0, 255, 0)
    red = (0, 0, 255)
    white = (255, 255, 255)
    black = (0, 0, 0)


@dataclass
class Font:
    face = cv2.FONT_HERSHEY_PLAIN
    scale = 1.0


def visualise_matches(stereo_pair, stereo_matcher, duration_rectification, duration_matching):
    """ This function will create an image that shows corresponding keypoints in two images."""

    if stereo_matcher.matches is None or not stereo_matcher.keypoints_left or not stereo_matcher.keypoints_right:
        return np.hstack((stereo_pair.left, stereo_pair.right))

    cv2.putText(stereo_pair.left, "LEFT", (10, 20), Font.face, Font.scale, Colours.black)
    cv2.putText(stereo_pair.right, "RIGHT", (10, 20), Font.face, Font.scale, Colours.black)
    vis_img = cv2.drawMatches(
        stereo_pair.left, stereo_matcher.keypoints_left,
        stereo_pair.right, stereo_matcher.keypoints_right,
        stereo_matcher.matches, None, flags=2)
    cv2.putText(vis_img, f"Rectification:  {round(duration_rectification)} ms", (10, 40), Font.face, Font.scale, Colours.red)
    cv2.putText(vis_img, f"Matching:  {round(duration_matching)} ms", (10, 60), Font.face, Font.scale, Colours.red)
    cv2.putText(vis_img, f"Number of matches:  {len(stereo_matcher.matches)}", (10, 80), Font.face, Font.scale, Colours.red)
    return vis_img


def visualise_depths(stereo_pair, pts_left, depths):
    """"""

    vis_depth = cv2.cvtColor(stereo_pair.left, cv2.COLOR_GRAY2BGR)

    marker_size = 5
    for pt, depth in zip(pts_left, depths):
        px = pt.astype(int)
        cv2.drawMarker(vis_depth, px, Colours.green, cv2.MARKER_CROSS, marker_size)
        cv2.putText(vis_depth, f"{depth:.2f}", px, Font.face, Font.scale, Colours.green)

    return vis_depth


def visualise_dense(dense_depth: np.ndarray, min_depth, max_depth, duration_dense):
    vis_dense_depth = (dense_depth / max_depth).astype(np.float32)
    vis_dense_depth = cv2.cvtColor(vis_dense_depth * 255, cv2.COLOR_GRAY2BGR).astype(np.uint8)
    vis_dense_depth = cv2.applyColorMap(vis_dense_depth, cv2.COLORMAP_JET)

    cv2.putText(vis_dense_depth, f"Dense:  {round(duration_dense)} ms", (10, 20), Font.face, Font.scale, Colours.white)
    cv2.putText(vis_dense_depth, f"Depth range:  [{min_depth:.2f}, {max_depth:.2f}]", (10, 40), Font.face, Font.scale, Colours.white)

    return vis_dense_depth
