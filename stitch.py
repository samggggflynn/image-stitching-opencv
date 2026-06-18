#!/usr/bin/env python3
"""
统一入口：现代化图像拼接工具

整合四个历史版本的功能，修复 image_stitching_new_fix_oom.py 中的逻辑错误：
1. 修复 ImageStitcher(max_dimension=...) 构造函数参数错误
2. 修复 args.crop > 0 布尔值比较错误
3. 真正实现 --max-size 参数防止 OOM

USAGE:
    python stitch.py --images images/scottsdale --output output.png --crop --max-size 2000
"""
import argparse
import logging
import sys
from pathlib import Path

import cv2

from modules import ImageStitcher, ImageLoader, ImageCropper, get_status_message

# 设置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="现代化图像拼接工具（模块化重构版）",
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
        help="是否裁剪到最大矩形区域（去除黑边）"
    )

    parser.add_argument(
        "--max-size",
        type=int,
        default=None,
        metavar="N",
        help="限制输入图像的最大边长（防止内存溢出OOM），例如 2000"
    )

    parser.add_argument(
        "--no-display",
        action="store_true",
        help="拼接完成后不弹出显示窗口"
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.3,
        help="匹配置信度阈值（0.1-1.0），降低可增加匹配成功率（默认 0.3）"
    )

    return parser.parse_args()


def main() -> int:
    """主函数，返回退出码"""
    args = parse_arguments()

    # 验证输入目录
    if not args.images.exists() or not args.images.is_dir():
        logger.error(f"输入目录无效: {args.images}")
        return 1

    # 确保输出目录存在
    args.output.parent.mkdir(parents=True, exist_ok=True)

    try:
        # 1. 加载图像（带尺寸限制，修复OOM）
        loader = ImageLoader(max_dimension=args.max_size)
        images = loader.load_from_directory(args.images)

        if len(images) < 2:
            logger.error("至少需要2张图像进行拼接")
            return 1

        # 2. 执行拼接
        stitcher = ImageStitcher(confidence_threshold=args.confidence)
        status, stitched = stitcher.stitch(images)

        if status != 0:
            logger.error(f"图像拼接失败: {get_status_message(status)}")
            return 1

        # 3. 可选裁剪（修正：布尔值判断）
        if args.crop:
            stitched = ImageCropper.crop_to_rectangle(stitched)

        # 4. 保存结果
        if not cv2.imwrite(str(args.output), stitched):
            logger.error(f"保存图像失败: {args.output}")
            return 1

        logger.info(f"拼接结果已保存到: {args.output}")

        # 5. 显示结果
        if not args.no_display:
            cv2.imshow("拼接结果", stitched)
            logger.info("按任意键关闭窗口...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return 0

    except Exception as e:
        logger.error(f"处理过程中发生错误: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
