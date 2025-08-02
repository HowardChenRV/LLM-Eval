import uuid

from .models import TestType


def generate_task_id(test_type: TestType) -> str:
    """
    生成任务ID（暂定为UUID v4）
    :param test_type: 测试类型
    :return: 任务ID字符串
    """
    _ = test_type
    return str(uuid.uuid4())
