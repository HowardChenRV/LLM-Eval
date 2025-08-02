import argparse
import json
from typing import Union
from abc import ABC, abstractmethod
from llm_eval.utils.configs import Env


class CLICommand(ABC):
    """
    Base class for command line tool.

    """

    @staticmethod
    @abstractmethod
    def define_args(parsers: argparse.ArgumentParser):
        raise NotImplementedError()

    @abstractmethod
    def execute(self):
        raise NotImplementedError()


def try_parse_json(value: str) -> Union[str, dict, None]:
    if value is None:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        if "{" in value:
            raise argparse.ArgumentTypeError(
                f"Invalid JSON: {value}. Hint: Use double quotes for JSON strings."
            )
        return value
    

def add_global_arguments(parser: argparse.ArgumentParser):
    """Add global arguments to the parser."""
    global_argument_group = parser.add_argument_group("global arguments options")
    global_argument_group.add_argument('--use-yaml', action='store_true', help='Use YAML file run benchmark.')
    global_argument_group.add_argument("--yaml-path", type=str, default="", help="Path to the benchmark case YAML file.")
    global_argument_group.add_argument("--env", type=str, choices=[e.value for e in Env], default=Env.TEST.value, help="Environment (DEV, TEST, PROD).")
    # extra arguments
    extra_argument_group = parser.add_argument_group("extra arguments options for you can customize")
    extra_argument_group.add_argument(
        "--extra-args",
        default="",
        type=try_parse_json,
        help="""Comma separated string or JSON formatted arguments for extra benchmark args, e.g. `ignore_eos=true,temperature=0.6,max_completion_tokens=128,stream=true` or '{"ignore_eos":true,"temperature":0.6,"max_completion_tokens":128,"stream":true}'.""",
    )
    extra_argument_group.add_argument(
        "--meta-data",
        default="",
        type=try_parse_json,
        help="""Comma separated string or JSON formatted arguments for test meta data, e.g. `tester=Jack,task_name=test,` or '{"tester":"Jack","task_name":"test"}'.""",
    )
