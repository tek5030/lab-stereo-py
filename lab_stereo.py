import cv2
import numpy as np
import timeit

from common_lab_utils import (CvStereoMatcherWrap, Size, StereoPair, StereoMatchingResult, hnormalized, homogeneous)
from visualisation import (Scene3D, visualise_dense, visualise_depths, visualise_matches)
from kitti_interface import KittiCamera
from stereo_calibration import StereoCalibration
from real_sense_stereo_camera import (RealSenseStereoCamera, LaserMode)
from anms import anms


def run_stereo_lab(cam, calibration):
    # TODO 1: If you want to use Kitti, choose this stereo source below.
    # Print camera info.
    print(f"camera:\n{cam}\ncalibration:\n{calibration}")

    # Set up the stereo matchers.
    detector = cv2.FastFeatureDetector_create()
    desc_extractor = cv2.ORB_create(nlevels=1)
    sparse_matcher = SparseStereoMatcher(detector, desc_extractor)
    dense_matcher = DenseStereoMatcher(calibration)

    # Set up the user interface.
    use_anms = False
    laser_on = False
    dense = False

    print("Press 'q' to quit.")
    print("Press 'a' to toggle adaptive non-maximal suppression.")
    print("Press 'd' to toggle dense processing.")
    if isinstance(cam, RealSenseStereoCamera):
        print("Press 'l' to toggle laser.")
    
    matching_win = "Stereo matching"
    depth_win = "Stereo depth"
    dense_win = "Dense disparity"

    cv2.namedWindow(matching_win, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(depth_win, cv2.WINDOW_AUTOSIZE)

    viewer_3d = Scene3D(calibration)

    # The main loop.
    while True:
        # Grab stereo images
        stereo_raw = cam.get_stereo_pair()
        if stereo_raw is None:
            break

        # Rectify images.
        start = timeit.default_timer()
        stereo_rectified = calibration.rectify(stereo_raw)
        end = timeit.default_timer()
        duration_rectification = (end - start) * 1000

        # Perform sparse matching.
        # TODO 2: Improve point correspondences in SparseStereoMatcher._extract_good_matches() below
        start = timeit.default_timer()
        match_result = sparse_matcher.match(stereo_rectified, use_anms)
        end = timeit.default_timer()
        duration_matching = (end - start) * 1000

        # Compute disparity for each match.
        # TODO 3: Compute the disparity for each match in compute_disparites() below.
        sparse_pts_left, sparse_disparities = compute_disparities(match_result)

        # Compute the corresponding depth for each disparity.
        # TODO 4: Compute the corresponding depth for each disparity in compute_depths() below.
        sparse_depths = compute_depths(sparse_disparities, calibration)

        # Compute 3D points for each disparity.
        # TODO 5: Compute 3D points in compute_3d_points() below.
        pts_3d = compute_3d_points(sparse_pts_left, sparse_disparities, calibration.q)

        # Dense stereo matching using OpenCV
        if dense:
            start = timeit.default_timer()
            dense_disparity, dense_depth = dense_matcher.match(stereo_rectified)
            end = timeit.default_timer()
            duration_dense = (end - start) * 1000

            # Visualise dense depth.
            vis_dense = visualise_dense(dense_depth, dense_matcher.min_depth, dense_matcher.max_depth, duration_dense)
            cv2.imshow(dense_win, vis_dense)

        # Visualise matched point correspondences
        match_image = visualise_matches(stereo_rectified, match_result, duration_rectification, duration_matching)
        cv2.imshow(matching_win, match_image)

        # Visualise sparse depth in meters for each point.
        vis_depth = visualise_depths(stereo_rectified, sparse_pts_left, sparse_depths)
        cv2.imshow(depth_win, vis_depth)

        # Visualise point cloude in 3D.
        viewer_3d.update(pts_3d)

        key = cv2.waitKey(50)
        if key == ord('q'):
            print("Bye")
            break
        elif key == ord('a'):
            use_anms = not use_anms
            print(f"Anms: {use_anms}")
        elif key == ord('d'):
            dense = not dense
            print(f"Dense: {dense}")
            if dense:
                cv2.namedWindow(dense_win, cv2.WINDOW_AUTOSIZE)
        elif key == ord('l') and isinstance(cam, RealSenseStereoCamera):
            laser_on = not laser_on
            cam.set_laser_mode(LaserMode.ON if laser_on else LaserMode.OFF)
            print(f"Laser: {laser_on}")


class SparseStereoMatcher:
    def __init__(self, detector, desc_extractor):
        self._detector = detector
        self._desc_extractor = desc_extractor
        self._matcher = cv2.BFMatcher_create(self._desc_extractor.defaultNorm())

    def match(self, stereo_pair: StereoPair, use_anms=False):
        """Detect and match keypoints in both images."""

        # Detect and describe features in the left image.
        keypoints_left = self._detector.detect(stereo_pair.left)
        if use_anms:
            img_size = Size.from_numpy_shape(stereo_pair.left.shape)
            keypoints_left = self._adaptive_non_maximal_suppression(keypoints_left, img_size)
        keypoints_left, query_descriptors = self._desc_extractor.compute(stereo_pair.left, keypoints_left)

        # Detect and describe features in the right image.
        keypoints_right = self._detector.detect(stereo_pair.right)
        if use_anms:
            img_size = Size.from_numpy_shape(stereo_pair.right.shape)
            keypoints_right = self._adaptive_non_maximal_suppression(keypoints_right, img_size)
        keypoints_right, train_descriptors = self._desc_extractor.compute(stereo_pair.right, keypoints_right)

        if not keypoints_left or not keypoints_right:
            return StereoMatchingResult()

        # Match features
        matches = self._matcher.match(query_descriptors, train_descriptors)

        # Extract good matches.
        good_matches = self._extract_good_matches(keypoints_left, keypoints_right, matches, epipolar_limit=1.5)

        return StereoMatchingResult(keypoints_left, keypoints_right, good_matches)

    @staticmethod
    def _adaptive_non_maximal_suppression(keypoints, img_size: Size, max_num=1000, max_ratio=0.7, tolerance=0.1):
        keypoints = sorted(keypoints, key=lambda x: x.response, reverse=True)

        num_to_retain = min(max_num, round(max_ratio * len(keypoints)))
        return anms.ssc(keypoints, num_to_retain, tolerance, img_size.width, img_size.height)

    @staticmethod
    def _extract_good_matches(keypoints_l, keypoints_r, matches, epipolar_limit):
        query_idx = [m.queryIdx for m in matches]
        train_idx = [m.trainIdx for m in matches]

        pts_l = np.array([k.pt for k in np.asarray(keypoints_l)[query_idx]])
        pts_r = np.array([k.pt for k in np.asarray(keypoints_r)[train_idx]])

        # TODO 2: Extract good matches by exploiting the stereo geometry.
        epipolar_is_ok = np.ones(len(matches)) > 0          # Dummy, do something more useful!
        disparity_is_positive = np.ones(len(matches)) > 0   # Dummy, do something more useful!

        good_matches = epipolar_is_ok & disparity_is_positive

        return np.asarray(matches)[good_matches]


def compute_disparities(match_result: StereoMatchingResult):
    if len(match_result.matches) <= 0:
        return np.array([]), np.array([])

    query_idx = [m.queryIdx for m in match_result.matches]
    train_idx = [m.trainIdx for m in match_result.matches]

    pts_l = np.array([k.pt for k in np.asarray(match_result.keypoints_left)[query_idx]])
    pts_r = np.array([k.pt for k in np.asarray(match_result.keypoints_right)[train_idx]])

    # TODO 3: Compute disparity.
    disparity = np.zeros(pts_l.shape[0])    # Dummy, compute disparity!

    return pts_l, disparity


def compute_depths(disparities: np.ndarray, calibration: StereoCalibration):
    if len(disparities) <= 0:
        return np.array([])

    fu = calibration.f
    bx = calibration.baseline

    # TODO 4: Compute depths.
    depths = np.zeros(disparities.shape)    # Dummy, compute depths!

    return depths


def compute_3d_points(pts_left: np.ndarray, disparities: np.ndarray, Q: np.ndarray):
    if len(disparities) <= 0:
        return np.array([])

    # TODO 5: Compute 3D points.
    pts_with_disparity = np.concatenate([pts_left.T, disparities[np.newaxis, :]], axis=0)
    pts_3d = np.array([])       # Dummy, compute 3D points!

    return pts_3d


class DenseStereoMatcher:
    def __init__(self, calibration: StereoCalibration, min_disparity=5, num_disparities=16*8, block_size=11):
        self._fu_bx = calibration.f * calibration.baseline

        self.min_disparity = min_disparity
        self.max_disparity = num_disparities + min_disparity

        self._dense_matcher = CvStereoMatcherWrap(cv2.StereoSGBM_create(
            minDisparity=min_disparity,
            numDisparities=num_disparities,
            blockSize=block_size,
            P1=8 * block_size ** 2,
            P2=32 * block_size ** 2,
            disp12MaxDiff=-1,
            preFilterCap=num_disparities,
            uniquenessRatio=0,
            speckleWindowSize=500,
            speckleRange=3,
            mode=cv2.StereoSGBM_MODE_HH
        ))

        # TODO 6.1: Compute min and max depths.
        self._min_depth = self.max_disparity    # Dummy, compute depth!
        self._max_depth = self.min_disparity    # Dummy, compute depth!

    def match(self, stereo_pair: StereoPair):
        dense_disparity = self._dense_matcher.compute(stereo_pair)
        dense_disparity[(dense_disparity < 0) | (dense_disparity > self.max_disparity)] = 0.

        # TODO 6.2: Compute depths.
        dense_depth = dense_disparity           # Dummy, compute depth!

        return dense_disparity, dense_depth

    @property
    def min_depth(self):
        return self._min_depth

    @property
    def max_depth(self):
        return self._max_depth


def kitti():
    import sys

    # Read paths from the command line arguments.
    cam = KittiCamera(*sys.argv[1:3])

    calibration = StereoCalibration.from_kitti(cam)
    return cam, calibration


def realsense():
    cam = RealSenseStereoCamera()
    calibration = StereoCalibration.from_realsense(cam)
    return cam, calibration


if __name__ == "__main__":
    # TODO 1: Choose stereo source
    run_stereo_lab(*realsense())
