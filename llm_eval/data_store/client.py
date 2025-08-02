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
    Data storage client
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
        Send test metadata (deprecated method, to be removed later)
        :param task_id: Task ID
        :param test_type: Test type
        :param meta_data: Metadata object
        :param kwargs: Optional parameters, when key=callback, specify the callback method for successful processing; when key=errback, specify the callback method for error handling
        :return: Asynchronous operation object
        """
        data = {'task_id': task_id, 'test_type': test_type.value, 'meta': meta_data}
        return self._send_data(META_KAFKA_TOPIC, data, kwargs.get('callback', None), kwargs.get('errback', None))

    @validate_call
    def send_test_data(self, task_id: str, test_type: TestType, data_type: DataType, data_time: datetime,
                       fields: Dict[str, Union[int, float, str, bool]],
                       tags: Dict[str, str] = None, **kwargs) -> FutureRecordMetadata:
        """
        Send test process data
        :param task_id: Task ID
        :param test_type: Test type
        :param data_type: Data type
        :param data_time: Time when data was generated
        :param fields: Store actual numeric data
        :param tags: Store metadata information and key-value pairs for identifying data
        :param kwargs: Optional parameters, when key=callback, specify the callback method for successful processing; when key=errback, specify the callback method for error handling
        :return: Asynchronous operation object
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
        Send test statistics data
        :param task_id: Task ID
        :param test_type: Test type
        :param statistic_dict: Statistical data key-value pairs
        :param kwargs: Optional parameters, when key=callback, specify the callback method for successful processing; when key=errback, specify the callback method for error handling
        :return:
        """
        timestamp_ms = int(time.time() * 1000)
        data = {'task_id': task_id, 'test_type': test_type.value, 'timestamp': timestamp_ms}
        data.update(statistic_dict)
        return self._send_data(STATISTIC_KAFKA_TOPIC, data, kwargs.get('callback', None), kwargs.get('errback', None))

    @validate_call
    def send_scene_data(self, task_id: str, test_type: TestType, scene_data: object, **kwargs) -> FutureRecordMetadata:
        """
        Send test scenario data
        :param task_id: Task ID
        :param test_type: Test type
        :param scene_data: Test scenario data object
        :param kwargs: Optional parameters, when key=callback, specify the callback method for successful processing; when key=errback, specify the callback method for error handling
        :return: Asynchronous operation object
        """
        data = {'task_id': task_id, 'test_type': test_type.value, 'scene': scene_data}
        return self._send_data(SCENE_KAFKA_TOPIC, data, kwargs.get('callback', None), kwargs.get('errback', None))

    @validate_call
    def send_conclusion_data(self, task_id: str, test_type: TestType, conclusion_data: object,
                             **kwargs) -> FutureRecordMetadata:
        """
        Send test conclusion data
        :param task_id: Task ID
        :param test_type: Test type
        :param conclusion_data: Metadata object
        :param kwargs: Optional parameters, when key=callback, specify the callback method for successful processing; when key=errback, specify the callback method for error handling
        :return: Asynchronous operation object
        """
        data = {'task_id': task_id, 'test_type': test_type.value, 'conclusion': conclusion_data}
        return self._send_data(CONCLUSION_KAFKA_TOPIC, data, kwargs.get('callback', None), kwargs.get('errback', None))

    @validate_call
    def send_evaluation_result(self, task_id: str, evaluation_result: EvaluationResult) -> FutureRecordMetadata:
        """
        Send model effect evaluation result data
        :param task_id: Task ID
        :param evaluation_result: Evaluation result object
        :return: FutureRecordMetadata Asynchronous processing result
        """
        return self.send_statistic_data(task_id, TestType.MODEL_INFERENCE_EVALUATION, asdict(evaluation_result))

    def upload_evaluation_detail(self, task_id: str, dataset_name: str, filenames: List[str], expire_day: int = 180) -> \
            Dict[str, str]:
        """
        Upload model effect evaluation detail files
        :param task_id: Task ID
        :param dataset_name: Dataset name
        :param filenames: Collection of local file paths
        :param expire_day: URL expiration days, default 180 days
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
        Download model effect evaluation dataset with specified name and version
        :param dataset_name: Dataset name
        :param version: Dataset version
        :param filepath: Local file path
        """
        for obj in self._get_evaluation_dataset_iterator(dataset_name, version):
            key = obj.key
            filename = os.path.basename(key)
            self._download_file(key, os.path.join(filepath, filename))

    def upload_evaluation_latest_dataset(self, dataset_name: str, filename: str, meta_data: Dict[str, any]):
        """
        Upload the latest version of evaluation dataset
        :param dataset_name: Dataset name
        :param filename: Local file path
        :param meta_data: Metadata dict
        """
        self._rename_latest_dataset(dataset_name)
        self._update_evaluation_dataset(dataset_name=dataset_name, version='latest', filename=filename,
                                        meta_data=meta_data)

    def download_evaluation_latest_dataset(self, dataset_name: str, filepath: str):
        """
        Download the latest model effect dataset under the specified name
        :param dataset_name: Dataset name
        :param filepath: Local file path
        """
        self.download_evaluation_dataset(dataset_name, 'latest', filepath)

    def get_evaluation_latest_dataset_version(self, dataset_name: str):
        """
        Get the version number of the latest evaluation dataset
        :param dataset_name: Dataset name
        :return: Version number
        """
        meta_data = self._get_evaluation_dataset_metadata(dataset_name, 'latest')
        if 'version' in meta_data:
            return meta_data['version']
        else:
            return None

    def _send_data(self, topic: str, data: object, callback: Callable = None,
                   errback: Callable = None) -> FutureRecordMetadata:
        """
        Send data to Kafka queue
        :param topic: Topic name
        :param data: Data content
        :param callback: Normal callback
        :param errback: Exception callback
        :return: Asynchronous object
        """
        future = self._producer.send(topic, data)
        if callback is not None:
            future.add_callback(callback)
        if errback is not None:
            future.add_errback(errback)
        return future
