import cv2
import numpy as np
from visual_odometry import VisualOdometry


def draw_coord_sys(coord_sys: np.ndarray, frame: np.ndarray, mat_rot: np.ndarray, trans: np.ndarray,
                   mat_cam: np.ndarray) -> np.ndarray:
    """
    Draw a coordinate system on a frame
    :param coord_sys: [4x3] 1 point on x, 1 point on y, 1 point on z, origin, expressed in the world frame
    :param frame: image to draw on
    :param mat_rot: rotation matrix. This matrix together with trans constitute the mapping from world frame to camera frame
    :param trans: translation vector
    :param mat_cam: camera's intrinsic matrix
    """
    # convert rotation matrix to rotation vector
    vec_rot, _ = cv2.Rodrigues(mat_rot)
    # project points onto current frame
    proj_pts, _ = cv2.projectPoints(coord_sys, vec_rot, trans, mat_cam, (0, 0, 0, 0))
    proj_pts = proj_pts.astype(int)
    frame = cv2.arrowedLine(frame, tuple(proj_pts[3].ravel()), tuple(proj_pts[0].ravel()), (255, 0, 0), 3)
    frame = cv2.arrowedLine(frame, tuple(proj_pts[3].ravel()), tuple(proj_pts[1].ravel()), (0, 255, 0), 3)
    frame = cv2.arrowedLine(frame, tuple(proj_pts[3].ravel()), tuple(proj_pts[2].ravel()), (0, 0, 255), 3)
    return frame


def main(video_path: str = ''):
    vo = VisualOdometry([684.169232, 680.105767, 365.163397, 292.358876])
    cap = cv2.VideoCapture('../Materials/armen.mp4')

    # define plane's local frame in 1st camera frame
    axes_size = .1
    ax_x = vo.c0_M_p[:, :3] @ np.array([[axes_size, 0, 0]]).T + vo.plane_origin_src
    ax_y = vo.c0_M_p[:, :3] @ np.array([[0, axes_size, 0]]).T + vo.plane_origin_src
    ax_z = vo.c0_M_p[:, :3] @ np.array([[0, 0, axes_size]]).T + vo.plane_origin_src
    plane_in_c0 = np.vstack((ax_x.T, ax_y.T, ax_z.T, vo.plane_origin_src.T))  # <4, 3>

    # to record video
    if video_path != '':
        width, height = int(cap.get(3)), int(cap.get(4))
        out = cv2.VideoWriter('vo_result.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (width, height))

    i = 0
    while cap.isOpened():
        _, frame = cap.read()
        if frame is None:
            break

        print('\n---------- Frame {} begins ----------'.format(i))

        incoming_M_c0 = vo.run(frame, update_src_frame=True)
        if incoming_M_c0.size > 0:
            # vo produces a solution
            frame = draw_coord_sys(plane_in_c0, frame, incoming_M_c0[:3, :3], incoming_M_c0[:3, 3], vo.camera_intrinsic)

        print('---------- Frame {} ends ----------\n'.format(i))

        cv2.imshow('current frame', frame)
        if video_path != '':
            out.write(frame)
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break
        i += 1

    cap.release()
    if video_path != '':
        out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
