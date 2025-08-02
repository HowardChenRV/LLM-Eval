import PyInstaller.__main__
import os

def build_exe():
    # Define packaging parameters
    params = [
        '--name=llm-eval',
        '--onefile',
        '--console',  # Replace --windowed with --console to support command line
        '--add-data=llm_eval;llm_eval',
        '--add-data=requirements;requirements',
        '--hidden-import=llm_eval.cli.serving_perf_eval',
        '--hidden-import=llm_eval.cli.lenovo_perf_eval',
        '--hidden-import=wandb_gql',
        '--hidden-import=wandb',
        '--hidden-import=wandb.sdk',
        '--hidden-import=wandb.apis',
        '--hidden-import=wandb.sdk.internal',
        '--hidden-import=wandb.integration.kfp',
        '--hidden-import=wandb_workspaces',
        '--hidden-import=kfp.components._python_op',
        '--hidden-import=sklearn',
        '--hidden-import=transformers',
        '--hidden-import=torch',
        '--hidden-import=aiohttp',
        '--hidden-import=datasets',
        '--hidden-import=modelscope',
        '--hidden-import=modelscope.hub.api',
        '--hidden-import=modelscope.msdatasets',
        '--hidden-import=modelscope.utils',
        '--hidden-import=faker',
        '--collect-all', 'wandb',
        '--collect-all', 'modelscope',
        '--collect-all', 'transformers',
        '--collect-all', 'kfp',
        '--collect-all', 'sklearn',
        '--collect-all', 'faker',
        'llm_eval/cli/main.py'
    ]
    
    # Execute packaging
    PyInstaller.__main__.run(params)

if __name__ == "__main__":
    build_exe()