import os
import sys
from enum import Enum
from typing import Optional, Union
from omegaconf import OmegaConf
from .logger import init_global_log

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "conf")


class Env(Enum):
    DEV = "development"
    TEST = "testing"
    PROD = "production"


def handle_arg_string(arg):
    if arg.lower() == "true":
        return True
    elif arg.lower() == "false":
        return False
    elif arg.isnumeric():
        return int(arg)
    try:
        return float(arg)
    except ValueError:
        return arg
    

def simple_parse_args_string(args_string: Optional[str]) -> dict:
    """
    Parses something like
        args1=val1,arg2=val2
    Into a dictionary
    """
    if args_string is None:
        return {}
    args_string = args_string.strip()
    if not args_string:
        return {}
    arg_list = [arg for arg in args_string.split(",") if arg]
    args_dict = {
        kv[0]: handle_arg_string("=".join(kv[1:]))
        for kv in [arg.split("=") for arg in arg_list]
    }
    return args_dict


class ConfigManager:
    
    def __init__(self, 
        env: str = Env.TEST.value,
        extra_args: Union[str, dict, None] = "",
        meta_data: Union[str, dict, None] = ""
    ):
        config_path = os.path.join(BASE_DIR, "config.yaml")
        config_path = os.path.abspath(config_path)
        
        print(f"Initialize config, env={env}")
        self.config = OmegaConf.create()
        config_yaml = OmegaConf.load(config_path)
        self.config = config_yaml.default
        
        # merge specific env config
        if env != Env.TEST.value:
            self.config = OmegaConf.merge(self.config, config_yaml[env])
            
        # merge test meta data and extra args
        test_meta = simple_parse_args_string(meta_data) if isinstance(meta_data, str) else meta_data if isinstance(meta_data, dict) else {}
        extra_args = simple_parse_args_string(extra_args) if isinstance(extra_args, str) else extra_args if isinstance(extra_args, dict) else {}
        
        self.config.test_meta = OmegaConf.create(test_meta)
        self.config.extra_args = OmegaConf.create(extra_args)
        
        
        print(f"Initialize config successed.")
        # print(OmegaConf.to_yaml(self.config))
        
        # Initialize logging level based on the config
        init_global_log(
            level=self.config.logging.level, 
            enable_save=self.config.logging.enable_save, 
            log_dir=self.config.logging.log_dir
        )
        
    
    def add_benchmark_by_yaml(self, case_path: str):
        """通过YAML文件添加配置
        
        Args:
            case_path: YAML配置文件路径
            
        Note:
            1. test_meta配置会赋给config.test_meta
            2. 其他所有配置会赋给config.benchmark
            3. YAML文件示例结构:
               benchmark_name: "serving_perf_eval"
               test_meta:
                 "key1": "value1"
                 "key2": "value2"
               backend: "MAAS"
               ...
        """
        if not case_path:
            sys.exit("If use yaml, case_path parameter is required.")
            
        case_path = os.path.abspath(case_path)
        if not os.path.isfile(case_path):
            sys.exit("Invalid path or file does not exist.")
        if not case_path.lower().endswith((".yaml", ".yml")):
            sys.exit("The file is not in YAML format.")
            
        config = OmegaConf.load(case_path)
        benchmark = OmegaConf.create()
        
        for key in config:
            if key == "test_meta":
                self.config.test_meta = OmegaConf.merge(self.config.test_meta, config[key])
            elif key == "extra_args":
                self.config.extra_args = OmegaConf.merge(self.config.extra_args, config[key])
            else:
                benchmark[key] = config[key]
        
        self.config.benchmark = benchmark
    

    def print_config(self):
        print(OmegaConf.to_yaml(self.config))

