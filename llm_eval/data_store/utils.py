import uuid

from .models import TestType


def generate_task_id(test_type: TestType) -> str:
    """
    Generate task ID (UUID v4)
    :param test_type: Test type
    :return: Task ID string
    """
    _ = test_type
    return str(uuid.uuid4())
