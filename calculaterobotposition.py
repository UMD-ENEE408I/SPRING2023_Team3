from pupil_apriltags import Detector
import cv2
import numpy as np
import time
import threading
import math


# Defined below is a class with different variables to calculate for the threading code
# Tag_data os a dictionary with the 8 apriltags around the arena, 1 through 8 rotating anticlockwise (accounts for position and orientation)
class Coordinates:
    def __init__(self):
        self.position = np.array(([0, 0, 0, 0]))
        self.theta = 0
        self.phi = 0
        self.checkpoint = np.array(([1, 7]))
        self.tag_data = {1: np.array(([[-1, 0, 0, 2, ],
                                       [0, 0, -1, 0],
                                       [0, -1, 0, 0],
                                       [0, 0, 0, 1]])),
                         2: np.array(([[-1, 0, 0, 4],
                                       [0, 0, -1, 0],
                                       [0, -1, 0, 0],
                                       [0, 0, 0, 1]])),
                         3: np.array(([[0, -1, 0, 6],
                                       [0, 0, -1, 2],
                                       [1, 0, 0, 0],
                                       [0, 0, 0, 1]])),
                         4: np.array(([[0, -1, 0, 6],
                                       [0, 0, -1, 4],
                                       [1, 0, 0, 0],
                                       [0, 0, 0, 1]])),
                         5: np.array(([[1, 0, 0, 4],
                                       [0, 0, -1, 6],
                                       [0, 1, 0, 0],
                                       [0, 0, 0, 1]])),
                         6: np.array(([[1, 0, 0, 2],
                                       [0, 0, -1, 6],
                                       [0, 1, 0, 0],
                                       [0, 0, 0, 1]])),
                         7: np.array(([[0, 1, 0, 0],
                                       [0, 0, -1, 4],
                                       [1, 0, 0, 0],
                                       [0, 0, 0, 1]])),
                         8: np.array(([[0, 1, 0, 0],
                                       [0, 0, -1, 2],
                                       [1, 0, 0, 0],
                                       [0, 0, 0, 1]]))}

        self.at_detector = Detector(
            families="tag36h11",
            nthreads=1,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0
        )

    # This function find the position of the camera in terms of the predefined coordinate plane
    def find_position_from_tag(self, K, detection):
        tag_size = 0.16 / 0.3048
        m_half_size = tag_size / 2

        marker_center = np.array((0, 0, 0))
        marker_points = [marker_center + (-m_half_size, m_half_size, 0), marker_center + (m_half_size, m_half_size, 0),
                         marker_center + (m_half_size, -m_half_size, 0),
                         marker_center + (-m_half_size, -m_half_size, 0)]
        _marker_points = np.array(marker_points)

        object_points = _marker_points
        image_points = detection.corners

        pnp_ret = cv2.solvePnP(object_points, image_points, K, distCoeffs=None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not pnp_ret[0]:
            raise Exception('Error solving PnP')

        r = pnp_ret[1]
        p = pnp_ret[2]

        return p.reshape((3,)), r.reshape((3,))

    def calculaterobotposition(self):
        vid = cv2.VideoCapture(0)

        tag_size = 0.16 / 0.3048  # tag size in meters, divided by ratio to have calculations in square titles (feet)

        while True:
            try:
                ret, img = vid.read()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray.astype(np.uint8)

                # The K matrix below is the result of running camera calibration
                # it has to be calibrated per camera model, this is calibrated for the camera located in Dylan's box
                K = np.array(
                    [[209.7613861666375, 0.0, 325.6130163782315], [0.0, 210.09468677064044, 249.80823959567041],
                     [0.0, 0.0, 1.0]])
                D = np.array([[0.5661790059116305], [0.2995215113127441], [-0.8056120138320512], [0.38069855704747346]])

                map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, (640, 480), cv2.CV_16SC2)
                undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR,
                                            borderMode=cv2.BORDER_CONSTANT)
                gray = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2GRAY)

                results = self.at_detector.detect(gray, estimate_tag_pose=False)

                # Values to store the position and rotation of nearest apriltag
                mintca = None

                for res in results:
                    pose = self.find_position_from_tag(K, res)
                    rot, jaco = cv2.Rodrigues(pose[1], pose[1])

                    pts = res.corners.reshape((-1, 1, 2)).astype(np.int32)
                    undistored_img = cv2.polylines(undistorted_img, [pts], isClosed=True, color=(0, 0, 255),
                                                   thickness=5)
                    cv2.circle(img, tuple(res.center.astype(np.int32)), 5, (0, 0, 255), -1)

                    # Calculates the matrix transform by combining pose and rot, and taking the inverse
                    rot_and_pose = [rot, pose[0].reshape((3, 1))]
                    rot_and_pose = np.hstack(rot_and_pose)
                    Transform_Camera_Apriltagk = (rot_and_pose, np.array([0, 0, 0, 1]))
                    Transform_Camera_Apriltagk = np.vstack(Transform_Camera_Apriltagk)
                    Transform_Apriltagk_Camera = np.linalg.inv(Transform_Camera_Apriltagk)

                    # Find tag ID
                    Transform_World_Tag = self.tag_data[res.tag_id]

                    # Get final transform matrix
                    Transform_World_Camera = np.matmul(Transform_World_Tag, Transform_Apriltagk_Camera)

                    # Get camera position
                    Position = np.matmul(Transform_World_Camera, [0, 0, 0, 1])

                    # Get camera rotation
                    Rotation = (Transform_World_Camera[:3, :3])
                    
                    #Transform 3x3 Rotation matrix into theta
                    Theta = math.atan2(Rotation[0, 2], Rotation[1,2])
                    
                    #Caluclate Phi
                    Vw = np.array(([0, 0]))
                    Vw[0] = self.checkpoint[0] - Position[0] 
                    Vw[1] = self.checkpoint[1] - Position[1]
                    Vw = Vw/np.linalg.norm(Vw)
                    Vr = np.array(([0, 0]))
                    Vr = np.array(([math.cos(Theta), math.sin(Theta)]))
                    Vr = Vr/np.linalg.norm(Vr)
                    w = np.matmul(Vw, Vr)
                    Phi = math.acos(w)
                   
                    # If two or more apriltags are detected, use the closest one
                    if (mintca is None or np.linalg.norm(pose[0]) < mintca):
                        self.position = Position
                        self.theta = Theta
                        self.phi = Phi
                        mintca = np.linalg.norm(pose[0])

                # To print the position and rotation of closest apriltag
                #print(self.position, "\n")
                #print(self.theta, "\n")
               

                cv2.imshow("undistorted", undistorted_img)
                cv2.waitKey(1)

            except KeyboardInterrupt:
                vid.release()
                cv2.destroyAllWindows()
                print('Exiting')
                exit(1)


if __name__ == '__main__':
    u = Coordinates()
    positionthread = threading.Thread(target = u.calculaterobotposition)
    positionthread.start()
    
    while(True):
    	#print(u.position, "\n")
    	#print(u.theta, "\n")
    	print(u.phi, "\n")

# guvcview

