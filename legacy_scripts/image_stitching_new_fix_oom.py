# USAGE
# python modern_image_stitching.py --images images/scottsdale --output output.png --crop

import cv2
import numpy as np
import argparse
from pathlib import Path
from typing import List, Tuple, Optional
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImageStitcher:
    """现代化的图像拼接器"""
    
    def __init__(self):
        """初始化拼接器"""
        try:
            self.stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)
        except AttributeError:
            # 兼容旧版本OpenCV
            self.stitcher = cv2.createStitcher()
    
    def load_images(self, image_dir: Path) -> List[np.ndarray]:
        """
        从目录加载图像
        
        Args:
            image_dir: 图像目录路径
            
        Returns:
            图像列表
        """
        # 支持常见图像格式
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # 获取所有图像文件并排序
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(image_dir.glob(f'*{ext}'))
            image_paths.extend(image_dir.glob(f'*{ext.upper()}'))
        
        image_paths = sorted(image_paths)
        
        if not image_paths:
            raise ValueError(f"在目录 {image_dir} 中未找到图像文件")
        
        logger.info(f"发现 {len(image_paths)} 张图像")
        
        images = []
        for image_path in image_paths:
            try:
                image = cv2.imread(str(image_path))
                if image is None:
                    logger.warning(f"无法读取图像: {image_path}")
                    continue
                images.append(image)
                logger.info(f"已加载: {image_path.name}")
            except Exception as e:
                logger.error(f"加载图像 {image_path} 时出错: {e}")
                continue
        
        if not images:
            raise ValueError("没有成功加载任何图像")
            
        return images
    
    def stitch_images(self, images: List[np.ndarray]) -> Tuple[int, Optional[np.ndarray]]:
        """
        拼接图像
        
        Args:
            images: 图像列表
            
        Returns:
            状态码和拼接结果
        """
        logger.info("开始拼接图像...")
        return self.stitcher.stitch(images)
    
    def crop_to_rectangle(self, stitched: np.ndarray) -> np.ndarray:
        """
        裁剪到最大矩形区域
        
        Args:
            stitched: 拼接后的图像
            
        Returns:
            裁剪后的图像
        """
        logger.info("正在裁剪到最大矩形区域...")
        
        # 添加边框
        stitched = cv2.copyMakeBorder(
            stitched, 2, 2, 2, 2, 
            cv2.BORDER_CONSTANT, (0, 0, 0)
        )
        
        # 转换为灰度图并二值化
        gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
        
        # 找到最大轮廓
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            logger.warning("未找到轮廓，返回原图像")
            return stitched
        
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 创建掩码
        mask = np.zeros(thresh.shape, dtype="uint8")
        x, y, w, h = cv2.boundingRect(largest_contour)
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
        
        # 迭代侵蚀找到最大内接矩形
        min_rect = mask.copy()
        sub = mask.copy()
        
        while cv2.countNonZero(sub) > 0:
            min_rect = cv2.erode(min_rect, None)
            sub = cv2.subtract(min_rect, thresh)
        
        # 找到最终轮廓并提取边界框
        contours, _ = cv2.findContours(
            min_rect, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            return stitched[y:y + h, x:x + w]
        else:
            logger.warning("裁剪失败，返回原图像")
            return stitched

def parse_arguments() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="现代化图像拼接工具",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "-i", "--images", 
        type=Path, 
        required=True,
        help="输入图像目录路径"
    )
    
    parser.add_argument(
        "-o", "--output", 
        type=Path, 
        required=True,
        help="输出图像路径"
    )
    
    parser.add_argument(
        "-c", "--crop", 
        action="store_true",
        help="是否裁剪到最大矩形区域"
    )
    
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="不显示结果图像"
    )
    
    return parser.parse_args()

def get_status_message(status: int) -> str:
    """获取状态码对应的错误信息"""
    status_messages = {
        0: "成功",
        1: "需要更多图像",
        2: "同质性检查失败", 
        3: "相机参数调整失败",
        4: "波形校正失败",
        5: "曝光补偿失败",
        6: "接缝查找失败",
        7: "接缝估计失败",
        8: "合成失败"
    }
    return status_messages.get(status, f"未知错误 (状态码: {status})")

def main():
    """主函数"""
    args = parse_arguments()
    
    # 验证输入目录
    if not args.images.exists():
        logger.error(f"输入目录不存在: {args.images}")
        return
    
    if not args.images.is_dir():
        logger.error(f"输入路径不是目录: {args.images}")
        return
    
    # 确保输出目录存在
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # 创建拼接器并加载图像
        max_size = getattr(args, 'max_size', 1000)  # 兼容处理
        stitcher = ImageStitcher(max_dimension=max_size)
        images = stitcher.load_images(args.images)
        
        if len(images) < 2:
            logger.error("至少需要2张图像进行拼接")
            return
        
        # 执行拼接
        status, stitched = stitcher.stitch_images(images)
        
        if status != 0:
            logger.error(f"图像拼接失败: {get_status_message(status)}")
            return
        
        # 可选裁剪
        if args.crop > 0:
            stitched = stitcher.crop_to_rectangle(stitched)
        
        # 保存结果
        success = cv2.imwrite(str(args.output), stitched)
        if success:
            logger.info(f"拼接结果已保存到: {args.output}")
        else:
            logger.error(f"保存图像失败: {args.output}")
            return
        
        # 显示结果
        if not args.no_display:
            cv2.imshow("拼接结果", stitched)
            logger.info("按任意键关闭窗口...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
    except Exception as e:
        logger.error(f"处理过程中发生错误: {e}")
        return

if __name__ == "__main__":
    main()
