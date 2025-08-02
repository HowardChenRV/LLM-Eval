import argparse
# import asyncio
from dataclasses import dataclass
from omegaconf import OmegaConf
from llm_eval.benchmarks.lenovo_perf_eval.benchmark_aipc_perf import run_benchmark
from llm_eval.cli.base import CLICommand, add_global_arguments
from llm_eval.utils.configs import ConfigManager

# for registry
import importlib
PLUGIN_REGISTRY = [
    # dataset
    "llm_eval.datasets.lenovo.lenovo_mmlu",
    # api
    "llm_eval.apis.lenovo_api",
]
for module in PLUGIN_REGISTRY:
    importlib.import_module(module)

BENCHMARK_NAME = "lenovo_perf_eval"

@dataclass
class LenovoPerfEvalArgument:
    benchmark_name: str = BENCHMARK_NAME
    port: int = 8060
    model: str = ""
    output_prefix: str = ""
    warmup_times: int = 1
    repetition: int = 5

def add_argument(parser: argparse.ArgumentParser):
    parser.add_argument('--port', type=int, default=8060,
                       help='Port for Lenovo inference server')
    parser.add_argument('--model', type=str, default="",
                       help='Tokenizer path')
    parser.add_argument('--output-prefix', type=str, default="",
                       help='Output file prefix')
    parser.add_argument('--warmup-times', type=int, default=1,
                       help='Number of warmup times (default: 1)')
    parser.add_argument('--repetition', type=int, default=5,
                       help='Number of repetitions (default: 5)')

class LenovoPerfEvalCMD(CLICommand):
    name = BENCHMARK_NAME

    def __init__(self, args):
        self.args = args

    @staticmethod
    def define_args(parsers: argparse.ArgumentParser):
        parser = parsers.add_parser(LenovoPerfEvalCMD.name)
        add_global_arguments(parser)
        add_argument(parser)
        parser.set_defaults(func=lambda args: LenovoPerfEvalCMD(args))

    def execute(self):
        config_manager = ConfigManager(env=self.args.env, extra_args=self.args.extra_args, meta_data=self.args.meta_data)
        
        if self.args.use_yaml:
            config_manager.add_benchmark_by_yaml(case_path=self.args.yaml_path)
        else:
            eval_args = LenovoPerfEvalArgument(
                port=self.args.port,
                model=self.args.model,
                output_prefix=self.args.output_prefix,
                warmup_times=self.args.warmup_times,
                repetition=self.args.repetition
            )
            config_manager.config.benchmark = OmegaConf.structured(eval_args)
        
        config_manager.print_config()
        # asyncio.run(run_benchmark(config_manager.config))
        run_benchmark(config_manager.config)