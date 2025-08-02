from llm_eval.data_store.client import DataClient, generate_task_id, TestType
from omegaconf import OmegaConf, DictConfig
from functools import wraps
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Union
import logging


def check_data_client(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if self._data_client is None:
            logging.debug("DataClient is None")
            return
        return func(self, *args, **kwargs)
    return wrapper


def convert_enums_to_names(data: dict) -> dict:
    for key, value in data.items():
        if isinstance(value, Enum):
            data[key] = value.name
    return data


class ReportTask:
    def __init__(self, cfg: DictConfig, test_type: TestType, task_id: str = None, **kwargs):
        self._test_meta = cfg.test_meta
        self._task_id = generate_task_id(test_type) if task_id is None else task_id
        self._test_type = test_type
        self._data_client = None
        
        kwargs_config = OmegaConf.create(kwargs)
        self._test_meta = OmegaConf.merge(self._test_meta, kwargs_config)

        if cfg.metrics.enable_report:
            try:
                data_client = DataClient(f"{cfg.metrics.kafka.host}:{cfg.metrics.kafka.port}")
                self._data_client = data_client
                logging.info(f"ReportTask created, task_id: {self._task_id}, test_type: {test_type}")
            except Exception as e:
                logging.error(f"Error while creating DataClient: {e}")
        else:
            logging.info("Data Report switch closed.")


    @check_data_client
    def produce_statistic_data(self, fields: Union[dataclass, dict], tags: Union[dataclass, dict] = None):
        tags_dict = tags if isinstance(tags, dict) else (asdict(tags) if tags is not None else {})
        fields_dict = fields if isinstance(fields, dict) else (asdict(fields) if fields is not None else {})
        
        statistic_dict = {**self._test_meta, **tags_dict, **fields_dict}
        statistic_dict = convert_enums_to_names(statistic_dict)
        
        # Filter out invalid data types
        allowed_types = (int, float, str, bool, type(None))
        filtered_dict = {
            k: v for k, v in statistic_dict.items()
            if isinstance(v, allowed_types)
        }
        if len(filtered_dict) != len(statistic_dict):
            invalid_keys = set(statistic_dict.keys()) - set(filtered_dict.keys())
            logging.info(f"Removed invalid statistic data types for keys: {invalid_keys}")
            statistic_dict = filtered_dict
        
        self._data_client.send_statistic_data(
            task_id = self._task_id,
            test_type = self._test_type,
            statistic_dict = statistic_dict
        )
        logging.info(f"Statistic data sent, task_id: {self._task_id}, test_type: {self._test_type}")
        
        