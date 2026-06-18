"""核心拼接模块 - 封装OpenCV Stitcher"""
import logging
from typing import List, Tuple, Optional
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ImageStitcher:
    """OpenCV 图像拼接器封装，支持自定义参数"""

    def __init__(self, mode: int = cv2.Stitcher_PANORAMA, 
                 confidence_threshold: float = 1.0):
        """
        初始化拼接器

        Args:
            mode: 拼接模式，默认为全景模式
            confidence_threshold: 置信度阈值（默认 1.0，降低可增加匹配成功率）
        """
        try:
            self.stitcher = cv2.Stitcher.create(mode)
            # 降低匹配阈值，允许更少特征点拼接
            self.stitcher.setRegistrationResol(0.6)
            self.stitcher.setPanoConfidenceThresh(confidence_threshold)
            logger.info(f"使用 cv2.Stitcher.create() 创建拼接器 (阈值={confidence_threshold})")
        except AttributeError:
            # 兼容旧版本 OpenCV 3.x
            self.stitcher = cv2.createStitcher()
            logger.info("使用 cv2.createStitcher() 创建拼接器（旧版本兼容）")

    def stitch(self, images: List[np.ndarray]) -> Tuple[int, Optional[np.ndarray]]:
        """
        执行图像拼接

        Args:
            images: 待拼接的图像列表（至少2张）

        Returns:
            (status, result): 状态码和拼接结果
        """
        logger.info(f"开始拼接 {len(images)} 张图像...")
        return self.stitcher.stitch(images)
