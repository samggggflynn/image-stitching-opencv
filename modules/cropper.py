"""图像裁剪模块 - 提取拼接结果的最大矩形有效区域"""
import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ImageCropper:
    """将拼接后的图像裁剪到最大内接矩形区域"""

    @staticmethod
    def crop_to_rectangle(stitched: np.ndarray) -> np.ndarray:
        """
        裁剪到最大矩形区域

        通过形态学操作找到拼接结果中的最大有效矩形区域，
        去除黑色边缘。

        Args:
            stitched: 拼接后的原始图像

        Returns:
            裁剪后的图像
        """
        logger.info("正在裁剪到最大矩形区域...")

        # 添加2像素黑色边框，确保边缘轮廓闭合
        stitched = cv2.copyMakeBorder(
            stitched, 2, 2, 2, 2,
            cv2.BORDER_CONSTANT, (0, 0, 0)
        )

        # 转换为灰度图并二值化：有效区域为255，黑色背景为0
        gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

        # 找到最大轮廓
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            logger.warning("未找到有效轮廓，返回原图像")
            return stitched

        largest_contour = max(contours, key=cv2.contourArea)

        # 创建初始掩码（外接矩形）
        mask = np.zeros(thresh.shape, dtype="uint8")
        x, y, w, h = cv2.boundingRect(largest_contour)
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

        # 迭代侵蚀找到最大内接矩形
        min_rect = mask.copy()
        sub = mask.copy()

        while cv2.countNonZero(sub) > 0:
            min_rect = cv2.erode(min_rect, None)
            sub = cv2.subtract(min_rect, thresh)

        # 提取最终裁剪区域
        contours, _ = cv2.findContours(
            min_rect, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            return stitched[y:y + h, x:x + w]

        logger.warning("裁剪失败，返回原图像")
        return stitched
