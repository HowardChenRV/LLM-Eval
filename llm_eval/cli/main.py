import argparse
from llm_eval.cli.serving_perf_eval import ServingPerfEvalCMD
from llm_eval.cli.lenovo_perf_eval import LenovoPerfEvalCMD


def run_command():
    parser = argparse.ArgumentParser("LLM-Eval Command Line tool", usage="llm-eval <command> [<args>]")

    subparsers = parser.add_subparsers()
    
    ServingPerfEvalCMD.define_args(subparsers)
    LenovoPerfEvalCMD.define_args(subparsers)
    
    args = parser.parse_args()
    
    if not hasattr(args, "func"):
        parser.print_help()
        exit(1)

    cmd = args.func(args)
    cmd.execute()


if __name__ == "__main__":
    run_command()