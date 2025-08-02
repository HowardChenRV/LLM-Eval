from data_store.client import DataClient, generate_task_id, TestType
from typing import Dict, List, Optional
from functools import wraps
from datetime import datetime
from typing import Dict
import logging
import yaml
import os
import subprocess
import json
import traceback
import argparse
import sys


def load_yaml_config(file_path) -> Dict:
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


def check_data_client(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if self._data_client is None:
            logging.debug("DataClient is None")
            return
        return func(self, *args, **kwargs)
    return wrapper


class ReportTask:
    def __init__(self, cfg: Dict, test_type: TestType, task_id: str = None):
        self._task_id = generate_task_id(test_type) if task_id is None else task_id
        self._test_type = test_type
        self._meta_info = cfg.get("meta_info", {})
        
        self._data_client = None
        
        try:
            data_client = DataClient(f"{cfg['metrics']['kafka']['host']}:{cfg['metrics']['kafka']['port']}")
            self._data_client = data_client
            logging.info(f"ReportTask created, task_id: {self._task_id}, test_type: {test_type}")
        except Exception as e:
            logging.error(f"Error while creating DataClient: {e}")

    @check_data_client
    def produce_statistic_data(self, fields: Dict):        
        statistic_dict = {**self._meta_info, **fields}
        
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


def run_llama_bench(
    model_path: str,
    llama_bench_path: str = "llama-bench.exe",
    prompt_tokens: str = "128,512",
    gen_tokens: str = "64,128",
    timeout: Optional[int] = 900,
) -> List[Dict]:
    """
    Run llama-bench with real-time output to both console and log file.
    """
    # Clean and validate parameters
    prompt_tokens = prompt_tokens.replace(" ", "")
    gen_tokens = gen_tokens.replace(" ", "")
    if not model_path.lower().endswith('.gguf'):
        raise ValueError(f"Model path must be a .gguf file, got: {model_path}")

    # Build the command
    cmd = [
        llama_bench_path,
        "-m", model_path,
        "-p", prompt_tokens,
        "-n", gen_tokens,
        "-o", "json"
    ]

    # Log the command being executed
    logging.info(f"Executing command: {' '.join(cmd)}")

    try:
        # Start the process with real-time output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            shell=True,
            bufsize=1,  # Line buffered
        )

        # Variables to capture output
        stdout_lines = []
        stderr_lines = []

        # Read output in real-time
        while True:
            # Read stdout
            stdout_line = process.stdout.readline()
            if stdout_line:
                stdout_lines.append(stdout_line)
                logging.info(stdout_line.strip())  # Log to both file and console
                print(stdout_line.strip())  # Print to console

            # Read stderr
            stderr_line = process.stderr.readline()
            if stderr_line:
                stderr_lines.append(stderr_line)
                logging.error(stderr_line.strip())  # Log to both file and console
                print(stderr_line.strip(), file=sys.stderr)  # Print to stderr

            # Check if process has completed
            if process.poll() is not None:
                # Read any remaining output after process ends
                for stdout_line in process.stdout:
                    stdout_lines.append(stdout_line)
                    logging.info(stdout_line.strip())
                    print(stdout_line.strip())

                for stderr_line in process.stderr:
                    stderr_lines.append(stderr_line)
                    logging.error(stderr_line.strip())
                    print(stderr_line.strip(), file=sys.stderr)
                break

        # Combine all output
        stdout = ''.join(stdout_lines)
        stderr = ''.join(stderr_lines)

        # Check return code
        if process.returncode != 0:
            raise subprocess.CalledProcessError(
                process.returncode, cmd, stdout, stderr
            )

        # Extract JSON from output
        json_start = stdout.find('[')
        json_end = stdout.rfind(']') + 1

        if json_start == -1 or json_end == 0:
            raise ValueError(
                f"Could not find JSON output in command result.\n"
                f"Full stdout:\n{stdout}\n"
                f"Full stderr:\n{stderr}"
            )

        json_str = stdout[json_start:json_end]

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Failed to parse JSON. Partial output:\n{json_str}\n"
                f"Full stdout:\n{stdout}\n"
                f"Full stderr:\n{stderr}",
                e.doc, e.pos,
            )

    except subprocess.CalledProcessError as e:
        error_msg = (
            f"Command failed with code {e.returncode}:\n"
            f"Command: {' '.join(e.cmd)}\n"
            f"STDOUT:\n{e.stdout}\n"
            f"STDERR:\n{e.stderr}\n"
        )
        logging.error(error_msg)
        raise subprocess.CalledProcessError(
            e.returncode, e.cmd, error_msg
        ) from e

    except subprocess.TimeoutExpired as e:
        error_msg = (
            f"Command timed out after {timeout} seconds:\n"
            f"Command: {' '.join(cmd)}\n"
            f"Partial stdout:\n{e.stdout}\n"
            f"Partial stderr:\n{e.stderr}\n"
        )
        logging.error(error_msg)
        raise subprocess.TimeoutExpired(
            cmd, timeout, output=error_msg
        ) from e

    except FileNotFoundError as e:
        error_msg = (
            f"Could not find executable or model file:\n"
            f"Executable: {llama_bench_path}\n"
            f"Model: {model_path}\n"
            f"Original error: {e}"
        )
        logging.error(error_msg)
        raise FileNotFoundError(error_msg) from e

    except Exception as e:
        error_msg = (
            f"Unexpected error: {type(e).__name__}: {str(e)}\n"
            f"Command: {' '.join(cmd)}\n"
            f"Traceback:\n{traceback.format_exc()}"
        )
        logging.error(error_msg)
        raise RuntimeError(error_msg) from e



def run_benchmark(config: Dict, llama_bench_path: str) -> bool:
    success = True
    logging.info("Running benchmark with the provided configuration.")
    
    test_case = config.get("test_case", {})
    if not test_case:
        logging.error("No test case configuration found.")
        return False
        
    model_list = test_case.get("model_list", [])
    for model in model_list:
        logging.info(f"Running benchmark for model: {model}")
        
        try:
            results = run_llama_bench(
                model_path=model["model_path"],
                llama_bench_path=llama_bench_path,
                prompt_tokens=test_case.get("prompt_tokens", "128,512"),
                gen_tokens=test_case.get("gen_tokens", "64,128")
            )
            logging.info(f"Benchmark completed for model {model['model_name']}")
            
            for result in results:
                result.update({
                    "model_name": model["model_name"],
                    "quantization_method": model.get("quantization_method", "Unknown"),
                    "spectype": model.get("spectype", "Unknown")
                })
                report_task = ReportTask(config, TestType.LLAMA_BENCH_PERFORMANCE)
                report_task.produce_statistic_data(result)
        except Exception as e:
            success = False
            logging.error(f"Benchmark FAILED for model {model.get('model_name', 'unknown')}: {str(e)}")
            logging.error(traceback.format_exc())  # 确保完整堆栈被记录
    return success


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description='Run llama-bench and process the output.')
        parser.add_argument('--llama-bench-path', default='llama-bench.exe', 
                         help='Path to llama-bench.exe (default: llama-bench.exe)')
        parser.add_argument('--branch', default=None, 
                        help='Git branch name (default: None)')
        parser.add_argument('--commit', default=None, 
                        help='Git commit hash (default: None)')
        args = parser.parse_args()
        
        # 初始化配置
        config = load_yaml_config("llama_bench_case.yaml")
        config["meta_info"].update({
            "branch": args.branch or "",
            "commit": args.commit or ""
        })
        
        # 初始化日志
        os.makedirs(config["logging"]["file_path"], exist_ok=True)
        log_file = os.path.join(
            config["logging"]["file_path"], 
            f"runtime_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        
        logging.basicConfig(
            level=config["logging"]["level"],
            format=config["logging"]["format"],
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file)
            ]
        )
        logging.getLogger("kafka").setLevel(logging.ERROR)
        # 执行测试
        all_success = run_benchmark(config, args.llama_bench_path)
        
        if not all_success:
            logging.error("Some benchmarks failed!")
            sys.exit(1)
            
        logging.info("All benchmarks completed successfully")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Fatal error in main execution: {str(e)}")
        logging.error(traceback.format_exc())
        sys.exit(2)

