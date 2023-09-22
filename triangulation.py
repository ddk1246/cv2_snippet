import numpy as np
import cv2


def get_exter_with_rt(R, T, is_square=False):
    """
    由 旋转和平移矩阵合成外参矩阵
    :param R: 3*3 旋转矩阵
    :param T: 3*1 平移矩阵
    :param is_square: 是否返回方阵
    :return:
    """
    R_T = np.concatenate([R, T], axis=1)
    if is_square:
        E = np.eye(4)
        E[:3] = R_T
        return E
    return R_T


def get_img_pos(p_w: np.ndarray,
                exter: np.ndarray,
                inter: np.ndarray,
                is_homogeneous=False):
    """
    p_img =  inter @ exter @ p_world

    通过内外参矩阵 获得空间点在图像坐标系下的位置
    :param p_w: 4*1 or 3*1
    :param exter: 3*4   or  4*4 ,
                 [R, T]     [R, T
                             0, 1]
    :param inter: 3*3
    :param is_homogeneous: 是否返回齐次化坐标
    :return: 2*1, [[x], [y]]
    """

    assert p_w.ndim == 2, f"Error dim{p_w.ndim} ,Coordinates should have 2 dim"
    assert p_w.shape[0] in [3, 4], f"Param p_w should 4*1 or 3*1, but get {p_w.shape}"
    assert exter.shape in ((3, 4), (4, 4))

    if p_w.shape[0] == 3:
        p_w = np.concatenate([p_w, np.ones((1, 1))], axis=0)  # 4*1
    if exter.shape[0] == 4:
        exter = exter[:3]

    p_camera = np.matmul(exter, p_w)
    p_img = np.matmul(inter, p_camera)
    p_img_homogeneous = p_img / p_img[2]
    if is_homogeneous:
        return p_img_homogeneous
    else:
        return p_img_homogeneous[:2]


if __name__ == '__main__':
    fx_left, fy_left = 100, 100
    cx_left, cy_left = 50, 50

    fx_right, fy_right = 100, 100
    cx_right, cy_right = 50, 50

    x, y, z = 6000, 3500, 1000

    # 定义左右相机的内参矩阵
    K_left = np.array([[fx_left, 0, cx_left],
                       [0, fy_left, cy_left],
                       [0, 0, 1]])

    K_right = np.array([[fx_right, 0, cx_right],
                        [0, fy_right, cy_right],
                        [0, 0, 1]])

    # 定义左右相机的旋转矩阵和平移矩阵
    R_left = np.array([[0.91812303, 0.39312509, 0.05002762],
                       [-0.33556185, 0.83835646, -0.42960061],
                       [-0.21082776, 0.37763885, 0.90163216]])

    T_left = np.array([[300],
                       [200],
                       [100]])

    E_left = get_exter_with_rt(R_left, T_left, is_square=True)  # 3 * 4

    R_right = np.array([[0.86787448, 0.49014195, 0.08096142],
                        [-0.38379768, 0.76500193, -0.51717636],
                        [-0.31542547, 0.41777136, 0.852041]])

    T_right = np.array([[1],
                        [3],
                        [2]])

    E_right = get_exter_with_rt(R_right, T_right, is_square=True)

    # 定义世界坐标系下的点
    P_world = np.array([[x],
                        [y],
                        [z],
                        [1]])

    # 左相机投影坐标
    P_left_image = get_img_pos(P_world, E_left, K_left, is_homogeneous=True)

    # 右相机投影
    P_right_image = get_img_pos(P_world, E_right, K_right, is_homogeneous=True)

    L_2_R = E_left @ np.linalg.inv(E_right)  # 4*4
    # R = R_left @ R_right.T
    # [t1, t2, t3] = (T_left - R_left @  R_right.T @ T_right)[:,0]

    R = L_2_R[:3, :3]

    [t1, t2, t3] = L_2_R[:3, 3]

    T_X = np.array([[0, -t3, t2],
                    [t3, 0, -t1],
                    [-t2, t1, 0]])

    line = P_left_image.T @ np.linalg.inv(K_left).T @ T_X @ R @ np.linalg.inv(K_right) @ P_right_image

    # 解算空间位置
    P_world_reconstructed_homogeneous = cv2.triangulatePoints(K_left @ E_left[:3]
                                                              , K_right @ E_right[:3]
                                                              , P_left_image[:2], P_right_image[:2])

    # 齐次化坐标
    P_world_reconstructed = P_world_reconstructed_homogeneous / P_world_reconstructed_homogeneous[3]

    # 打印结果
    print("对极几何: ", line)
    print("左相机投影坐标：", P_left_image)
    print("右相机投影坐标：", P_right_image)
    print("空间位置解算结果：", P_world_reconstructed[:3])
