"""图像加载模块 - 支持OOM防护的尺寸限制"""
import logging
from pathlib import Path
from typing import List, Optional
import cv2
import numpy as np

logger = logging.getLogger(__name__)

# 支持的图像扩展名
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}


class ImageLoader:
    """图像加载器，支持目录批量加载和尺寸限制"""

    def __init__(self, max_dimension: Optional[int] = None):
        """
        初始化加载器

        Args:
            max_dimension: 图像最大边长限制，超过则等比缩放。
                          设为 None 则不限制（可能引发OOM）。
        """
        self.max_dimension = max_dimension

    def find_images(self, image_dir: Path) -> List[Path]:
        """
        在目录中查找所有支持的图像文件

        Args:
            image_dir: 图像目录路径

        Returns:
            排序后的图像路径列表
        """
        image_paths = []
        for ext in SUPPORTED_EXTENSIONS:
            image_paths.extend(image_dir.glob(f'*{ext}'))
            image_paths.extend(image_dir.glob(f'*{ext.upper()}'))

        return sorted(image_paths)

    def resize_if_needed(self, image: np.ndarray) -> np.ndarray:
        """
        如果图像超过最大边长限制，则等比缩放

        Args:
            image: 原始图像

        Returns:
            缩放后的图像（如果需要）
        """
        if self.max_dimension is None:
            return image

        h, w = image.shape[:2]
        max_side = max(h, w)

        if max_side <= self.max_dimension:
            return image

        # 计算缩放比例
        scale = self.max_dimension / max_side
        new_w = int(w * scale)
        new_h = int(h * scale)

        logger.info(f"图像尺寸 {w}x{h} 超过限制，缩放到 {new_w}x{new_h}")
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def load_single(self, image_path: Path) -> Optional[np.ndarray]:
        """
        加载单张图像，失败时返回None

        Args:
            image_path: 图像文件路径

        Returns:
            加载后的图像数组，失败则为None
        """
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                logger.warning(f"无法读取图像: {image_path}")
                return None

            image = self.resize_if_needed(image)
            logger.info(f"已加载: {image_path.name}")
            return image

        except Exception as e:
            logger.error(f"加载图像 {image_path} 时出错: {e}")
            return None

    def load_from_directory(self, image_dir: Path) -> List[np.ndarray]:
        """
        从目录加载所有图像

        Args:
            image_dir: 图像目录路径

        Returns:
            图像列表

        Raises:
            ValueError: 目录不存在、不是目录、或未找到图像
        """
        if not image_dir.exists():
            raise ValueError(f"输入目录不存在: {image_dir}")

        if not image_dir.is_dir():
            raise ValueError(f"输入路径不是目录: {image_dir}")

        image_paths = self.find_images(image_dir)

        if not image_paths:
            raise ValueError(f"在目录 {image_dir} 中未找到图像文件")

        logger.info(f"发现 {len(image_paths)} 张图像")

        images = []
        for path in image_paths:
            image = self.load_single(path)
            if image is not None:
                images.append(image)

        if not images:
            raise ValueError("没有成功加载任何图像")

        return images
