"""状态码与错误信息映射模块"""

# 拼接状态码映射表
STATUS_MESSAGES = {
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


def get_status_message(status: int) -> str:
    """
    根据状态码获取对应的错误信息

    Args:
        status: OpenCV Stitcher 返回的状态码

    Returns:
        状态码对应的中文描述，未知状态码则返回原始信息
    """
    return STATUS_MESSAGES.get(status, f"未知错误 (状态码: {status})")
