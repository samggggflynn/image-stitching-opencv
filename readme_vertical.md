# 竖向图像拼接工具使用指南

## 概述

`vertical_stitch.py` 是一个统一的竖向图像拼接工具，整合了以下算法：

| 模式            | 算法                   | 适用场景   |
| ------------- | -------------------- | ------ |
| `auto`        | 自动选择最佳算法             | 默认推荐   |
| `feature`     | SIFT + FLANN + 单应性矩阵 | 高精度拼接  |
| `translation` | 平移矩阵 + 手动偏移          | 简单竖向拼接 |
| `simple`      | 固定重叠比例               | 快速拼接   |

## 基本用法

### 自动选择算法（推荐）

```bash
python vertical_stitch.py \
    -t /path/to/top_image.png \
    -b /path/to/bottom_image.png \
    -o /path/to/output.png
```

### 指定拼接模式

```bash
# 特征匹配模式（高精度）
python vertical_stitch.py \
    -t top.png \
    -b bottom.png \
    -o result.png \
    -m feature

# 平移矩阵模式（支持手动调整）
python vertical_stitch.py \
    -t top.png \
    -b bottom.png \
    -o result.png \
    -m translation \
    -s -200

# 简单垂直拼接（固定重叠）
python vertical_stitch.py \
    -t top.png \
    -b bottom.png \
    -o result.png \
    -m simple
```

## 参数说明

| 参数                | 类型     | 必填 | 说明                                           |
| ----------------- | ------ | -- | -------------------------------------------- |
| `-t/--top`        | Path   | ✅  | 上面图像的路径                                      |
| `-b/--bottom`     | Path   | ✅  | 下面图像的路径                                      |
| `-o/--output`     | Path   | ✅  | 输出图像的路径                                      |
| `-m/--mode`       | String | ❌  | 拼接模式：`auto`/`feature`/`translation`/`simple` |
| `-s/--shift`      | Float  | ❌  | 垂直偏移量（仅 translation 模式有效）                    |
| `--no-auto-shift` | Flag   | ❌  | 禁用自动检测偏移量                                    |

## 针对 featured\_sample1 的使用示例

### 测试数据路径

```
images/featured_sample1/tobestiched/
├── up_back_rgb_Color.png    # 上图
└── down_back_rgb_Color.png  # 下图
```

### 命令示例

```bash
# 进入项目目录
cd /home/zjt/dev/On_Git_Projects/image-stitching-opencv

# 1. 自动模式（推荐）
python vertical_stitch.py \
    -t images/featured_sample1/tobestiched/up_back_rgb_Color.png \
    -b images/featured_sample1/tobestiched/down_back_rgb_Color.png \
    -o result_auto.png

# 2. 特征匹配模式
python vertical_stitch.py \
    -t images/featured_sample1/tobestiched/up_back_rgb_Color.png \
    -b images/featured_sample1/tobestiched/down_back_rgb_Color.png \
    -o result_feature.png \
    -m feature

# 3. 平移矩阵模式（自动检测偏移）
python vertical_stitch.py \
    -t images/featured_sample1/tobestiched/up_back_rgb_Color.png \
    -b images/featured_sample1/tobestiched/down_back_rgb_Color.png \
    -o result_trans_auto.png \
    -m translation

# 4. 平移矩阵模式（手动调整偏移量）
python vertical_stitch.py \
    -t images/featured_sample1/tobestiched/up_back_rgb_Color.png \
    -b images/featured_sample1/tobestiched/down_back_rgb_Color.png \
    -o result_trans_manual.png \
    -m translation \
    -s -100 \
    --no-auto-shift

# 5. 简单垂直拼接
python vertical_stitch.py \
    -t images/featured_sample1/tobestiched/up_back_rgb_Color.png \
    -b images/featured_sample1/tobestiched/down_back_rgb_Color.png \
    -o result_simple.png \
    -m simple
```

## 偏移量说明

`-s/--shift` 参数用于调整垂直偏移量：

| 值         | 含义                          |
| --------- | --------------------------- |
| `-s -200` | 相机向上移动 200 像素 → 上图相对于下图向下偏移 |
| `-s 100`  | 相机向下移动 100 像素 → 上图相对于下图向上偏移 |
| 默认        | 自动检测或使用默认重叠比例               |

## 模式选择建议

| 场景     | 推荐模式                 | 说明           |
| ------ | -------------------- | ------------ |
| 一般情况   | `auto`               | 自动选择最佳算法     |
| 需要高精度  | `feature`            | 基于 SIFT 特征匹配 |
| 特征点较少  | `translation`        | 平移矩阵更稳健      |
| 快速拼接   | `simple`             | 固定重叠比例       |
| 需要手动调整 | `translation` + `-s` | 手动指定偏移量      |

## 与旧脚本的对应关系

| 旧脚本                                    | 新命令                                                        |
| -------------------------------------- | ---------------------------------------------------------- |
| `stitch.py --confidence 0.1`           | `vertical_stitch.py -m feature`                            |
| `stitch_vertical.py img1 img2 out.png` | `vertical_stitch.py -t img1 -b img2 -o out.png -m feature` |
| `stitch_vertical_enhanced.py`          | `vertical_stitch.py -m feature`                            |
| `stitch_vertical_final.py --mode auto` | `vertical_stitch.py -m auto`                               |
| `stitch_vertical_translate.py -s -300` | `vertical_stitch.py -m translation -s -300`                |

## 模块化调用示例

```python
from modules import AutoVerticalStitcher, TranslationStitcher

# 方法 1：使用自动拼接器
stitcher = AutoVerticalStitcher()
success, result = stitcher.stitch(img_top, img_bottom, mode='auto')

# 方法 2：使用平移矩阵拼接器
stitcher = TranslationStitcher()
success, result, estimated_shift = stitcher.stitch(
    img_top, img_bottom,
    shift_y=-200,
    auto_detect=False
)
```

## 输出示例

```
INFO:__main__:输入图像: up_back_rgb_Color.png ((1074, 896, 3)) + down_back_rgb_Color.png ((1080, 896, 3))
INFO:modules.vertical:自动选择拼接算法...
INFO:modules.vertical:特征点: top=925, bottom=1448
INFO:modules.vertical:初步匹配: 28
INFO:modules.vertical:竖向过滤后: 15
INFO:modules.vertical:有效匹配点(RANSAC): 9
INFO:modules.vertical:特征匹配拼接完成: 1291 x 1786
INFO:modules.vertical:使用特征匹配拼接
INFO:__main__:结果已保存到: result.png
INFO:__main__:输出尺寸: 1291 x 1786
```

## 故障排除

### 问题：特征匹配失败

**原因**：图像特征点太少或重叠区域不足

**解决方案**：

```bash
# 尝试平移矩阵模式
python vertical_stitch.py -t top.png -b bottom.png -o result.png -m translation

# 或使用简单拼接
python vertical_stitch.py -t top.png -b bottom.png -o result.png -m simple
```

### 问题：拼接结果有明显接缝

**原因**：重叠区域融合不佳

**解决方案**：

```bash
# 尝试调整偏移量
python vertical_stitch.py -t top.png -b bottom.png -o result.png -m translation -s -150
```

## 项目结构

```
image-stitching-opencv/
├── vertical_stitch.py           # 命令行工具入口
└── modules/
    ├── __init__.py              # 模块导出
    ├── vertical.py              # 竖向拼接核心模块
    │   ├── BaseVerticalStitcher    # 基类
    │   ├── SimpleStitcher          # 简单拼接
    │   ├── FeatureBasedStitcher    # 特征匹配拼接
    │   ├── TranslationStitcher     # 平移矩阵拼接
    │   └── AutoVerticalStitcher    # 自动选择拼接器
    └── ...                      # 其他模块
```

