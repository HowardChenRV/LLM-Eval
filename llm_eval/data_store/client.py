import json
import logging
import os
import time
from dataclasses import asdict
from datetime import datetime
from typing import Callable, Union, List

from kafka import KafkaProducer
from kafka.producer.future import FutureRecordMetadata
from pydantic import validate_call

from .const import *
from .models import *
from .utils import *

logger = logging.getLogger(__name__)


class DataClient:
    """
    数据存储客户端
    """

    def __init__(self, kafka_server: str = "8.140.201.231:9094"):
        self._producer = KafkaProducer(
            bootstrap_servers=[kafka_server],
            security_protocol=SECURITY_PROTOCOL,
            sasl_mechanism=SASL_MECHANISM,
            sasl_plain_username=SASL_USERNAME,
            sasl_plain_password=SASL_PASSWORD,
            value_serializer=lambda m: json.dumps(m).encode('utf-8'),
        )

    @validate_call
    def send_meta_data(self, task_id: str, test_type: TestType, meta_data: object, **kwargs) -> FutureRecordMetadata:
        """
        发送测试元数据（废弃方法，后续删除）
        :param task_id: 任务ID
        :param test_type: 测试类型
        :param meta_data: 元数据对象
        :param kwargs: 可选参数，key=callback时，指定处理成功时的回调方法；key=errback时，指定发生错误时的回调方法
        :return: 异步操作对象
        """
        data = {'task_id': task_id, 'test_type': test_type.value, 'meta': meta_data}
        return self._send_data(META_KAFKA_TOPIC, data, kwargs.get('callback', None), kwargs.get('errback', None))

    @validate_call
    def send_test_data(self, task_id: str, test_type: TestType, data_type: DataType, data_time: datetime,
                       fields: Dict[str, Union[int, float, str, bool]],
                       tags: Dict[str, str] = None, **kwargs) -> FutureRecordMetadata:
        """
        发送测试过程数据
        :param task_id: 任务ID
        :param test_type: 测试类型
        :param data_type: 数据类型
        :param data_time: 数据产生的时间
        :param fields: 存储实际的数值数据
        :param tags: 存储元数据信息和标识数据的键值对
        :param kwargs: 可选参数，key=callback时，指定处理成功时的回调方法；key=errback时，指定发生错误时的回调方法
        :return: 异步操作对象
        """
        timestamp = int(data_time.timestamp() * 10 ** 9)
        data = {'task_id': task_id, 'test_type': test_type.value, 'data_type': data_type.value, 'timestamp': timestamp,
                'fields': fields, 'tags': tags}
        return self._send_data(DATA_KAFKA_TOPIC, data, kwargs.get('callback', None), kwargs.get('errback', None))

    @validate_call
    def send_statistic_data(self, task_id: str, test_type: TestType,
                            statistic_dict: Dict[str, Union[int, float, str, bool, None]],
                            **kwargs) -> FutureRecordMetadata:
        """
        发送测试统计数据
        :param task_id: 任务ID
        :param test_type: 测试类型
        :param statistic_dict: 统计数据键值对
        :param kwargs: 可选参数，key=callback时，指定处理成功时的回调方法；key=errback时，指定发生错误时的回调方法
        :return:
        """
        timestamp_ms = int(time.time() * 1000)
        data = {'task_id': task_id, 'test_type': test_type.value, 'timestamp': timestamp_ms}
        data.update(statistic_dict)
        return self._send_data(STATISTIC_KAFKA_TOPIC, data, kwargs.get('callback', None), kwargs.get('errback', None))

    @validate_call
    def send_scene_data(self, task_id: str, test_type: TestType, scene_data: object, **kwargs) -> FutureRecordMetadata:
        """
        发送测试场景数据
        :param task_id: 任务ID
        :param test_type: 测试类型
        :param scene_data: 测试场景数据对象
        :param kwargs: 可选参数，key=callback时，指定处理成功时的回调方法；key=errback时，指定发生错误时的回调方法
        :return: 异步操作对象
        """
        data = {'task_id': task_id, 'test_type': test_type.value, 'scene': scene_data}
        return self._send_data(SCENE_KAFKA_TOPIC, data, kwargs.get('callback', None), kwargs.get('errback', None))

    @validate_call
    def send_conclusion_data(self, task_id: str, test_type: TestType, conclusion_data: object,
                             **kwargs) -> FutureRecordMetadata:
        """
        发送测试结论数据
        :param task_id: 任务ID
        :param test_type: 测试类型
        :param conclusion_data: 元数据对象
        :param kwargs: 可选参数，key=callback时，指定处理成功时的回调方法；key=errback时，指定发生错误时的回调方法
        :return: 异步操作对象
        """
        data = {'task_id': task_id, 'test_type': test_type.value, 'conclusion': conclusion_data}
        return self._send_data(CONCLUSION_KAFKA_TOPIC, data, kwargs.get('callback', None), kwargs.get('errback', None))

    @validate_call
    def send_evaluation_result(self, task_id: str, evaluation_result: EvaluationResult) -> FutureRecordMetadata:
        """
        发送模型效果评估的结果数据
        :param task_id: 任务ID
        :param evaluation_result: 评估结果对象
        :return: FutureRecordMetadata 异步处理的结果
        """
        return self.send_statistic_data(task_id, TestType.MODEL_INFERENCE_EVALUATION, asdict(evaluation_result))

    def upload_evaluation_detail(self, task_id: str, dataset_name: str, filenames: List[str], expire_day: int = 180) -> \
            Dict[str, str]:
        """
        上传模型效果评估详情文件
        :param task_id: 任务ID
        :param dataset_name: 数据集名称
        :param filenames: 本地文件路径集合
        :param expire_day: URL过期天数，默认180天
        """
        now = datetime.now()
        file_urls = {}
        for filename in filenames:
            name = os.path.basename(filename)
            if name:
                key = f'inference_evaluation/detail/{dataset_name}/{now:%y%m%d%H%M}-{task_id}/{name}'
                try:
                    self._upload_file(key, filename)
                    file_urls[filename] = self._generate_share_url(key, expire_day)
                except Exception as e:
                    logger.error(f"Upload file {filename} error: {e}", exc_info=True)
        return file_urls

    def download_evaluation_dataset(self, dataset_name: str, version: str, filepath: str):
        """
        下载指定名称和版本的模型效果评测数据集
        :param dataset_name: 数据集名称
        :param version: 数据集版本
        :param filepath: 本地文件路径
        """
        for obj in self._get_evaluation_dataset_iterator(dataset_name, version):
            key = obj.key
            filename = os.path.basename(key)
            self._download_file(key, os.path.join(filepath, filename))

    def upload_evaluation_latest_dataset(self, dataset_name: str, filename: str, meta_data: Dict[str, any]):
        """
        上传最新版本的评测数据集
        :param dataset_name: 数据集名称
        :param filename: 本地文件路径
        :param meta_data: 元数据dict
        """
        self._rename_latest_dataset(dataset_name)
        self._update_evaluation_dataset(dataset_name=dataset_name, version='latest', filename=filename,
                                        meta_data=meta_data)

    def download_evaluation_latest_dataset(self, dataset_name: str, filepath: str):
        """
        下载指定名称下最新的模型效果数据集
        :param dataset_name: 数据集名称
        :param filepath:  本地文件路径
        """
        self.download_evaluation_dataset(dataset_name, 'latest', filepath)

    def get_evaluation_latest_dataset_version(self, dataset_name: str):
        """
        获取最新版本评测数据集的版本号
        :param dataset_name:  数据集名称
        :return: 版本号
        """
        meta_data = self._get_evaluation_dataset_metadata(dataset_name, 'latest')
        if 'version' in meta_data:
            return meta_data['version']
        else:
            return None

    def _send_data(self, topic: str, data: object, callback: Callable = None,
                   errback: Callable = None) -> FutureRecordMetadata:
        """
        发送数据到kafka队列中
        :param topic: topic名称
        :param data: 数据内容
        :param callback: 正常回调
        :param errback: 异常回调
        :return: 异步对象
        """
        future = self._producer.send(topic, data)
        if callback is not None:
            future.add_callback(callback)
        if errback is not None:
            future.add_errback(errback)
        return future
