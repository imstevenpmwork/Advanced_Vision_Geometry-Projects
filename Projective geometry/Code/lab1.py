import numpy as np
import cv2
from matplotlib import pyplot as plt


def euclidean_trans(theta, tx, ty):
    return np.array([
        [np.cos(theta), -np.sin(theta), tx],
        [np.sin(theta), np.cos(theta), ty],
        [0, 0, 1]
    ])

filename = '../Materials/fig1_6c.jpg'
img = cv2.imread(filename)

plt.imshow(img, cmap='gray')
pts = np.asarray(plt.ginput(4, timeout=-1))
plt.show()

print('chosen coord: ', pts)  # each row is a point

plt.plot(*zip(*pts), marker='o', color='r', ls='')
plt.imshow(img)
plt.show()

points = np.array([
    [1053.01298701,  460.10606061],
    [1364.7012987,   641.92424242],
    [1057.34199134,  867.03246753],
    [ 745.65367965,  641.92424242]
])

'''
Affine rectification
'''
print('\n-------- Task 1: Affine rectification --------')
pts_homo = np.concatenate((points, np.ones((4, 1))), axis=1)  # convert chosen pts to homogeneous coordinate
print('Task 1.1: Identify image of the line at inf on projective plane')
hor_0 = np.cross(pts_homo[0],pts_homo[1])
hor_1 = np.cross(pts_homo[2],pts_homo[3])
pt_ideal_0 = np.cross(hor_0, hor_1)
pt_ideal_0 /= pt_ideal_0[-1]  # normalize
print('@Task 1.1: first ideal point: ', pt_ideal_0)

ver_0 = np.cross(pts_homo[0],pts_homo[3])
ver_1 = np.cross(pts_homo[1],pts_homo[2])
pt_ideal_1 = np.cross(ver_0, ver_1)
pt_ideal_1 /= pt_ideal_1[-1]
print('@Task 1.1: second ideal point: ', pt_ideal_1)

l_inf = np.cross(pt_ideal_0,pt_ideal_1)
l_inf /= l_inf[-1]
print('@Task 1.1: line at infinity: ', l_inf)

print('Task 1.2: Construct the projectivity that affinely rectify image')
H = np.array([[1,0,0],[0,1,0],[l_inf[0],l_inf[1],l_inf[2]]])
print('@Task 1.2: image of line at inf on affinely rectified image: ', (np.linalg.inv(H).T @ l_inf.reshape(-1, 1)).squeeze())

H_E = euclidean_trans(np.deg2rad(0), 50, 250)

affine_img = cv2.warpPerspective(img, H_E @ H, (img.shape[1], img.shape[0]))

affine_pts = np.transpose((H_E @ H).dot(np.transpose(pts_homo)))
for i in range(affine_pts.shape[0]):
    affine_pts[i] /= affine_pts[i, -1]
    print('\t',affine_pts[i])

plt.plot(*zip(*affine_pts[:, :-1]), marker='o', color='r', ls='')
plt.imshow(affine_img)
plt.show()
print('-------- End of Task 1 --------\n')

'''
#Task 2: Metric rectification
'''

print('\n-------- Task 2: Metric rectification --------')
print('Task 2.1: transform 4 chosen points from projective image to affine image')
aff_hor_0 = np.cross(affine_pts[0],affine_pts[1])
aff_hor_1 = np.cross(affine_pts[2],affine_pts[3])

aff_ver_0 = np.cross(affine_pts[0],affine_pts[3])
aff_ver_1 = np.cross(affine_pts[1],affine_pts[2])

aff_hor_0 /= aff_hor_0[-1]
aff_hor_1 /= aff_hor_1[-1]
aff_ver_0 /= aff_ver_0[-1]
aff_ver_1 /= aff_ver_1[-1]
# Make new orthogonal constraint between the diagonals
aff_diag_0 = np.cross(affine_pts[0],affine_pts[2])
aff_diag_1 = np.cross(affine_pts[1],affine_pts[3])
print('@Task 2.1: first chosen point coordinate')
print('\t\t on projective image: ', pts_homo[0])
print('\t\t on affine image: ', affine_pts[0])

print('Task 2.2: construct constraint matrix C to find vector s')
C0 = np.array([aff_hor_0[0]*aff_ver_0[0],aff_hor_0[0]*aff_ver_0[1] + aff_hor_0[1]*aff_ver_0[0],aff_hor_0[1]*aff_ver_0[1]])

#C1 = np.array([aff_hor_1[0]*aff_ver_1[0],aff_hor_1[0]*aff_ver_1[1] + aff_hor_1[1]*aff_ver_1[0],aff_hor_1[1]*aff_ver_1[1]])
C1 = np.array([aff_diag_0[0]*aff_diag_1[0],aff_diag_0[0]*aff_diag_1[1] + aff_diag_0[1]*aff_diag_1[0],aff_diag_0[1]*aff_diag_1[1]])

C = np.vstack([C0, C1])
print('@Task 2.2: constraint matrix C:\n', C)

print('Task 2.3: Find s by looking for the kernel of C (hint: SVD)')
[usvd, ssvd, svdvh]= np.linalg.svd(C)
s = np.transpose(svdvh)[:, -1]
print('@Task 2.3: s = ', s)
print('@Task 2.3: C @ s = \n', C @ s.reshape(-1, 1))
mat_S = np.array([
    [s[0], s[1]],
    [s[1], s[2]],
])
print('@Task 2.3: matrix S:\n', mat_S)

print('Task 2.4: Find the projectivity that do metric rectificaiton')
E, Q = np.linalg.eig(mat_S)
K = Q.dot(np.sqrt(np.diag(E)))
K_inv = np.linalg.inv(K)
H = np.array([[K_inv[0][0],K_inv[0][1],0],[K_inv[1][0],K_inv[1][1],0],[0,0,1]])

aff_dual_conic = np.array([
    [s[0], s[1], 0],
    [s[1], s[2], 0],
    [0, 0, 0]
])

print('@Task 2.3: image of dual conic on metric rectified image: ', H @ aff_dual_conic @ H.T)

H_E = euclidean_trans(np.deg2rad(0), -250, -900) @ euclidean_trans(np.deg2rad(-20),0,0)
H_fin = H_E @ H
eucl_img = cv2.warpPerspective(affine_img, H_fin, (img.shape[1], img.shape[0]))

eucl_pts = (H_fin @ affine_pts.T).T
for i in range(eucl_pts.shape[0]):
    eucl_pts[i] /= eucl_pts[i, -1]

plt.plot(*zip(*eucl_pts[:, :-1]), marker='o', color='r', ls='')
plt.imshow(eucl_img)
plt.show()

