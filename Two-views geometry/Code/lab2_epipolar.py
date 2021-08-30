import numpy as np
import cv2
import matplotlib.pyplot as plt 


def normalize_transformation(points: np.ndarray) -> np.ndarray:
    """
    Compute a similarity transformation matrix that translate the points such that
    their center is at the origin & the avg distance from the origin is sqrt(2)
    :param points: <float: num_points, 2> set of key points on an image
    :return: (sim_trans <float, 3, 3>)
    """
    center = np.array([np.mean(points[:,0],axis=0),np.mean(points[:,1],axis=0)])  # TODO: find center of the set of points by computing mean of x & y
    # dist = np.array([np.sqrt((points[:,0] - center[0])**2 + (points[:,1] - center[1])**2)])  # TODO: matrix of distance from every point to the origin, shape: <num_points, 1>
    dist = np.linalg.norm(points - center, axis=1)
    s = np.sqrt(2)/np.mean(dist)  # TODO: scale factor the similarity transformation = sqrt(2) / (mean of dist)
    sim_trans = np.array([
        [s,     0,      -s * center[0]],
        [0,     s,      -s * center[1]],
        [0,     0,      1]
    ])
    return sim_trans


def homogenize(points: np.ndarray) -> np.ndarray:
    """
    Convert points to homogeneous coordinate
    :param points: <float: num_points, num_dim>
    :return: <float: num_points, 3>
    """
    return np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)


# read image & put them in grayscale
img1 = cv2.imread('../Materials/chapel00.png', 0)  # queryImage
img2 = cv2.imread('../Materials/chapel01.png', 0)  # trainImage

# detect kpts & compute descriptor
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# match kpts
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

# organize key points into matrix, each row is a point
query_kpts = np.array([kp1[m.queryIdx].pt for m in matches]).reshape((-1, 2))  # shape: <num_pts, 2>
train_kpts = np.array([kp2[m.trainIdx].pt for m in matches]).reshape((-1, 2))  # shape: <num_pts, 2>

# normalize kpts
T_query = normalize_transformation(query_kpts)  # get the similarity transformation for normalizing query kpts
normalized_query_kpts = (T_query @ homogenize(query_kpts).T).T # TODO: apply T_query to query_kpts to normalize them

T_train = normalize_transformation(train_kpts)  # get the similarity transformation for normalizing train kpts
normalized_train_kpts = (T_train @ homogenize(train_kpts).T).T  # TODO: apply T_train to train_kpts to normalize them

print(T_query)
print(T_train)
# construct homogeneous linear equation to find fundamental matrix
a1 = (normalized_train_kpts[:,0]*normalized_query_kpts[:,0]).reshape(-1, 1)
a2 = (normalized_train_kpts[:,0]*normalized_query_kpts[:,1]).reshape(-1, 1)
a3 = normalized_train_kpts[:,0].reshape(-1, 1)
a4 = (normalized_train_kpts[:,1]*normalized_query_kpts[:,0]).reshape(-1, 1)
a5 = (normalized_train_kpts[:,1]*normalized_query_kpts[:,1]).reshape(-1, 1)
a6 = normalized_train_kpts[:,1].reshape(-1, 1)
a7 = normalized_query_kpts[:,0].reshape(-1, 1)
a8 = normalized_query_kpts[:,1].reshape(-1, 1)
a9 = np.ones((len(normalized_train_kpts),1))
A = np.concatenate((a1,a2,a3,a4,a5,a6,a7,a8,a9),axis=1) # TODO: construct A according to Eq.(3) in lab subject

# TODO: find vector f by solving A f = 0 using SVD
# hint: perform SVD of A using np.linalg.svd to get u, s, vh (vh is the transpose of v)
# hint: f is the last column of v
u,s,vh = np.linalg.svd(A)
f = vh.T[:,-1]  # TODO: find f

# arrange f into 3x3 matrix to get fundamental matrix F
F = f.reshape(3, 3)
print('rank F: ', np.linalg.matrix_rank(F))  # should be = 3

# TODO: force F to have rank 2
# hint: perform SVD of F using np.linalg.svd to get u, s, vh
# hint: set the smallest singular value of F to 0
# hint: reconstruct F from u, new_s, vh
u,s,vh = np.linalg.svd(F)
index = np.where(s == np.amin(s))
s[-1] = 0
F = u @ np.diag(s) @ vh

assert np.linalg.matrix_rank(F) == 2, 'Fundamental matrix must have rank 2'

# TODO: de-normlaize F
# hint: last line of Algorithme 1 in the lab subject
F = T_train.T @ F @ T_query
F_gt = np.loadtxt('../Materials/chapel.00.01.F')

Fransac, mask= cv2.findFundamentalMat(query_kpts, train_kpts, cv2.FM_RANSAC)

print('ERROR OF F COMPUTED BY 8-POINT ALGORITHM:')
print(F - F_gt)
print('*****')
print('ERROR OF F COMPUTED BY CV RANSAC:')
print(Fransac - F_gt)

## 1.1.1 Practical

# Load F
F= np.loadtxt('../Materials/chapel.00.01.F')

# Get the real epipole from F
u,s,vh= np.linalg.svd(F.T)
e= vh.T[:,-1]
e /= e[-1]

# Choose two correspondances points
plt.imshow(img1, cmap='gray')
x= plt.ginput(2)
x= np.asarray(x)

# Homogenize the selected points
one= np.ones((1,2))
x= np.concatenate((x,one.T),axis=1)

# Compute the epipolar lines by l = F x
l= F @ x.T

# Compute the epipole from the epipolar lines computed
ec= np.cross(l[:,0],l[:,1])
ec /= ec[-1]

print('****')
print('REAL EPIPOLE: ', e)
print('COMPUTED EPIPOLE: ', ec)

# Visualize the epipolar lines computed
x2 = np.array([0, 500])

a2, b2, c2 = l[:,0].ravel()
y2 = -(x2*a2 + c2) / b2
a3, b3, c3 = l[:,1].ravel()
y3 = -(x2*a3 + c3) / b3

img2=cv2.imread('../Materials/chapel01.png')
image=cv2.line(img2,(x2[0],int(np.around(y2.T[0]))),(x2[-1],int(np.around(y2.T[-1]))),(0,255,0),(1))
image=cv2.line(img2,(x2[0],int(np.around(y3.T[0]))),(x2[-1],int(np.around(y3.T[-1]))),(0,255,0),(1))
cv2.imshow('Epipolar lines',image)

# PRESS A KEY TO FINISH INSTEAD OF CLOSING THE WINDOW
cv2.waitKey(0)
