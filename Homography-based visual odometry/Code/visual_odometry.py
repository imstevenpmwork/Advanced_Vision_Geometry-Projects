import numpy as np
import cv2
from typing import List, Tuple


def homogenized(x: np.ndarray) -> np.ndarray:
    """
    Convert a matrix whose rows represent a coordinate (2D or 3D) into homogeneous form
    :param x: <num_points, num_dimension>
    """
    assert len(x.shape) == 2, 'input must be a matrix'
    return np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)


class VisualOdometry:
    def __init__(self, camera_intrinsic: List):
        """
        Constructor of visual odometry class
        :param camera_intrinsic: [px, py, u0, v0]
        """
        self.print_info = True
        self.camera_intrinsic = np.array([
            [camera_intrinsic[0],   0,                      camera_intrinsic[2]],
            [0,                     camera_intrinsic[1],    camera_intrinsic[3]],
            [0,                     0,                      1]
        ])
        self.inv_K = np.linalg.inv(self.camera_intrinsic)  # cache inv of camera intrinsic to increase speed

        self.orb = cv2.ORB_create()  # key pts detector
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  # key pts matcher
        self.src_kpts: List[cv2.KeyPoint] = []  # key pts in source frame
        self.src_desc: List[np.ndarray] = []  # descriptors of key pts in source frame
        self.incoming_kpts: List[cv2.KeyPoint] = []  # key pts in incoming frame
        self.incoming_desc: List[np.ndarray] = []  # descriptors of key pts in incoming frame

        self.homography_min_correspondences = 10  # need at least 4, but 10 provides better result
        self.homography_ransac_threshold = 5  # reprojection error threshold used in RANSAC

        self.plane_d_src = -1  # distance from plane to src frame
        # guess of plane's pose in the 1st camera's frame assuming plane & 1st camera have the same x-axis
        alpha = np.deg2rad(0)  # to tune to get nicer visualization
        self.c0_M_p = np.array([
            [1,     0,                  0,                      -0.1],
            [0,     np.cos(alpha),      -np.sin(alpha),         -0.1],
            [0,     np.sin(alpha),      np.cos(alpha),          0.7],
        ])
        self.plane_normal_src = self.c0_M_p[:, 2].reshape(-1, 1)  # plane's normal vector expressed in src frame
        self.plane_origin_src = self.c0_M_p[:, 3].reshape(-1, 1)  # origin of plane frame expressed in src frame

        self.src_M_c0 = np.eye(4)  # mapping from 1st camera frame to src frame

    def find_matches(self, incoming_frame: np.ndarray) -> List[cv2.DMatch]:
        """
        Find matches between key pts in incoming frame and those in src frame
        :param incoming_frame: <int: height, width, 3>
        :return: matches
        """
        # convert incoming frame to gray scale
        gray = cv2.cvtColor(incoming_frame, cv2.COLOR_BGR2GRAY) # TODO

        # if this is the first frame, assign computed kpts to src and return an empty list
        if not self.src_kpts:
            self.src_kpts, self.src_desc = self.orb.detectAndCompute(gray,None) # TODO
            return []
        else:
            # find matches between incoming kpts and src kpts
            self.incoming_kpts, self.incoming_desc = self.orb.detectAndCompute(gray,None)  # TODO
            matches = self.matcher.match(self.src_desc,self.incoming_desc)  # TODO
            return matches

    def compute_relative_transform(self, matches: List[cv2.DMatch], update_src_frame: bool) -> \
            Tuple[np.ndarray, np.ndarray]:
        """
        Compute the transformation that maps points from src frame to incoming frame
        :param matches: list of matched kpts between incoming frame and src frame
        :param update_src_frame: whether to replace src frame by incoming frame by the end of computation
        :return: (incoming_M_src <float: 4, 4>, normal <float: 3, 1>)
        """
        if len(matches) < self.homography_min_correspondences:
            print('\t[WARN] Not enough correspondences to compute homography')
            return np.array([]), np.array([])

        # extract matched key pts & put them in the order of the "matches" list
        src_pts = np.array([self.src_kpts[m.queryIdx].pt for m in matches]).reshape((-1, 2))  # <num_pts, 2>
        incoming_pts = np.array([self.incoming_kpts[m.trainIdx].pt for m in matches]).reshape((-1, 2))  # <num_pts, 2>

        # compute homography
        mat_H, mask = cv2.findHomography(src_pts,incoming_pts, cv2.RANSAC, self.homography_ransac_threshold) # TODO

        # get inliers (with respect to the homography found above) in src_pts
        inliers = list(map(bool, mask.ravel().tolist()))
        src_pts = src_pts[inliers, :]  # <num_inliers, 2>
        src_pts_homo = homogenized(src_pts)  # convert src_pts into homogeneous coordinate <num_inliers, 3>
        # compute normalized coordinate of inliers
        src_pts_normalized = (self.inv_K @ src_pts_homo.T).T  # TODO: result should have the shape of <num_inliers, 3>

        # decompose homography to get (R, t, n)
        num_candidates, rots, trans, normals = cv2.decomposeHomographyMat(mat_H, self.camera_intrinsic)  # TODO
        if self.print_info:
            print('decomposition of homography yields {} candidates'.format(num_candidates))
            if rots:
                print('\t rots ({}), rots[0] is {}'.format(type(rots), rots[0].shape))
                print('\t trans ({}), trans[0] is {}'.format(type(trans), trans[0].shape))
                print('\t normals ({}), normals[0] is {}'.format(type(normals), normals[0].shape))

        if num_candidates > 1:
            #
            # PRUNE CANDIDATE LEADING TO AT LEAST 1 MATCHED KEYPOINT HAS NEGATIVE DEPTH
            #
            # for each candidate, compute depth for evey inlier & prune the candidate if >= 1 inlier has negative depth
            pruned_candidate_indices = []
            for i, n in enumerate(normals):
                # compute ratio of d/z for every inlier
                d_over_z = src_pts_normalized @ n # TODO: result should be an array of <float: num_inliers>
                # check if any ratio is negative
                is_valid = np.all(d_over_z > 0)  # TODO: True if every inlier has positive depth, False otherwise
                if not is_valid:
                    pruned_candidate_indices.append(i)
                    if self.print_info:
                        print('candidate {}, num pts have negative z: {}'.format(i, np.sum(d_over_z < 0)))

            # prune invalid candidate
            for i in reversed(pruned_candidate_indices):
                # delete candidates from the highest index to the smallest
                del rots[i]
                del trans[i]
                del normals[i]
            if self.print_info and pruned_candidate_indices:
                print('\t after pruning, {} candidates left'.format(len(rots)))

            #
            # CHOOSE CANDIDATE HAS n CLOSER TO THE **CURRENT** ESTIMATION IN CASE HAVE 2 CANDIDATE LEFT
            #
            if len(rots) > 1:
                assert len(rots) == 2, 'After pruning solution gives negative z, still have more than 1 candidate'
                angle_0 = np.arccos(np.dot(normals[0].T,self.plane_normal_src)) # TODO: compute angle between 1st candidate of normal vector with self.plane_normal_src
                angle_1 = np.arccos(np.dot(normals[1].T,self.plane_normal_src)) # TODO: compute angle between 2nd candidate of normal vector with self.plane_normal_src
                del_idx = 0 if angle_0 > angle_1 else 1
                del rots[del_idx]
                del trans[del_idx]
                del normals[del_idx]

        # check if any candidate survive after pruning
        if rots:
            # check if plane's distance to src frame has been initialized, if not compute it
            if self.plane_d_src < 0:
                if self.print_info:
                    print('initialize plane distance to src frame')
                self.plane_d_src = normals[0].T @ self.plane_origin_src  # TODO: compute distance from the 1st camera frame C0 to the plane
                assert self.plane_d_src > 0, 'plane distance to src frame must be positive'

            # scale the translation using plane's distance to src frame
            trans[0] = trans[0]*self.plane_d_src  # TODO: scale the translation vector with self.plane_d_src

            # build transformation from src frame to incoming frame
            # TODO create the transformation matrix from src camera frame to
            incoming_M_src = np.eye(4)
            incoming_M_src[:3, :3] = rots[0]
            incoming_M_src[:3,3] = trans[0].flatten() # 3x1
            # TODO: (continue) incoming camera frame using trans[0] and rots[0]

            if update_src_frame:
                # replace src key pts & descriptors with those of incoming frame
                self.src_kpts = self.incoming_kpts  # TODO
                self.src_desc = self.incoming_desc  # TODO

                # map the plane's normal from src frame to incoming frame
                self.plane_normal_src = rots[0] @ self.plane_normal_src  # TODO

                #
                # COMPUTE PLANE'S DISTANCE TO INCOMING FRAME
                #
                # find depth of every inliers (w.r.t homography found above)
                d_over_z = src_pts_normalized @ normals[0]  # TODO: result should be an array of <float: num_inliers>
                depth = 1/d_over_z * self.plane_d_src # TODO: compute depth of inlier using d and normalize coordinate, result should be
                # TODO: (continue) an array of <float: num_inliers,>

                # find 3D coordinate in src frame of these inliers
                src_pts_3d = src_pts_normalized * depth  # TODO: compute 3d coordinate in src camera frame for every inlier,
                # TODO: (continue) result should be a matrix of shape <num_inliers, 3>

                # map these points to incoming frame
                incoming_pts_3d = (rots[0] @ src_pts_3d.T).T + trans[0].T # TODO: result should have shape <num_inliers, 3>

                # compute new d by averaging projection of every point onto plane's normal vector
                self.plane_d_src = np.mean(incoming_pts_3d @ normals[0]) # TODO

            return incoming_M_src, normals[0]
        else:
            # no candidate survives, return empty matrix
            return np.array([]), np.array([])

    def run(self, incoming_frame: np.ndarray, update_src_frame: bool = False) -> np.ndarray:
        """
        Main function for visual odometry which computes the mapping from the 1st camera frame to incoming frame
        :param incoming_frame: <np.uint8: height, width, 3>
        :param update_src_frame: whether to update src frame
        :return: incoming_M_c0 <np.float: 4, 4>
        """
        matches = self.find_matches(incoming_frame)
        if not matches:
            # there are no matches between key pts in incoming frame & src frame
            return np.array([])

        incoming_M_src, normal = self.compute_relative_transform(matches, update_src_frame)

        if incoming_M_src.size == 0:
            # there is no solution, due to lack of matches or all candidates are pruned
            return np.array([])
        else:
            # has solution
            # compute transformation from c0 to incoming frame
            incoming_M_c0 = incoming_M_src @ self.src_M_c0
            if self.print_info:
                print('rot: \n', incoming_M_c0[:3, :3])
                print('trans: \n', incoming_M_c0[:3, 3])
                print('normal: \n', normal.flatten())
            if update_src_frame:
                # update the mapping from 1st camera frame (c0) to src (by replacing src with incoming)
                self.src_M_c0 = incoming_M_c0  # TODO
            return incoming_M_c0
