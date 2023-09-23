import mmcv
import numpy as np
import cv2
from dataclasses import dataclass


def order_points(pts: np.ndarray):
    """
    排序坐标顺序

    :param pts: 4*2
    :return:    0, 1
                3, 2
    """
    pts = np.array(pts, dtype=np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)

    s = np.sum(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    d = np.diff(pts, axis=1)  # 后减前
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]

    return rect


def sort_points(src_points):
    """
    给定 四个角点， 返回图像横置后的 角点和宽高
    :param src_points:  4*2
                        0, 1
                        2, 3
    :return: 左上 右上 右下 左下角点， 宽 , 高 (宽 >= 高)
    """

    rect_points = order_points(src_points)
    width = np.mean([np.linalg.norm(rect_points[0] - rect_points[1]),
                     np.linalg.norm(rect_points[2] - rect_points[3]), ])

    height = np.mean([np.linalg.norm(rect_points[1] - rect_points[2]),
                      np.linalg.norm(rect_points[3] - rect_points[0]), ])

    # should height << width
    if height > width:
        rect_points = rect_points[[1, 2, 3, 0]]
        height, width = width, height

    return rect_points, width, height


def get_out_points(dst_width, dst_height, margin):
    out_width = dst_width + 2 * margin
    out_height = dst_height + 2 * margin

    dst_points = margin + np.array([[0, 0],
                                    [dst_width - 1, 0],
                                    [dst_width - 1, dst_height - 1],
                                    [0, dst_height - 1]], dtype=np.float32)
    return dst_points, out_width, out_height


@dataclass
class Transform:
    margin: int = 3
    dst_width: int = 360
    dst_height: int = 10

    def __post_init__(self):
        self.margin = int(self.margin)
        self.dst_width = int(self.dst_width)
        self.dst_height = int(self.dst_height)

    def get_perspective_region(self,
                               src_img: np.ndarray,
                               src_points: np.ndarray,
                               is_force_size=False
                               ):
        """
        获得目标区域的仿射变换图
        :param src_img: H*W*C
        :param src_points: 4*[x, y]
        :param is_force_size: True 强制使用自定义大小, False 使用图像本身大小
        :return: ROI
        """

        sorted_points, w, h = sort_points(src_points)
        if is_force_size:
            dst_points, out_width, out_height = get_out_points(self.dst_width, self.dst_height, self.margin)
        else:
            dst_points, out_width, out_height = get_out_points(int(w), int(h), self.margin)

        M = cv2.getPerspectiveTransform(sorted_points, dst_points)
        out_img = cv2.warpPerspective(src_img, M, (out_width, out_height), )
        return out_img

    def get_source_region(self,
                          src_img: np.ndarray,
                          src_points: np.ndarray,
                          ):
        min_x, min_y = np.min(src_points, axis=0)
        max_x, max_y = np.max(src_points, axis=0)

        roi_img = src_img[min_y:max_y, min_x:max_x]
        if (max_y - min_y) > (max_x - min_x):
            roi_img = mmcv.imrotate(roi_img, 90, auto_bound=True)
        return roi_img

    def post_process(self, image):
        # 图像锐化增强
        pass


if __name__ == '__main__':
    aff = Transform()
    name, img_path, points = iman.get_img_and_points_by_name('608')[0]

    img = mmcv.imread(img_path)
    # img = mmcv.imrescale(img, 0.2)

    aff_img = aff.get_source_region(img, points * 5)
    mmcv.imshow(aff_img, 'affine')
