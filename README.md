## Installation
### Install from Source
1. Download the source code
   ```shell
   git clone https://github.com/HowardChenRV/LLM-Eval.git
   ```

2. Create a conda environment (optional)
   We recommend using conda to manage your environment:
   ```shell
   # It is recommended to use Python 3.10 ~ 3.11
   conda create -n llm-eval python=3.10
   # Activate the conda environment
   conda activate llm-eval
   ```

3. Install dependencies
   ```shell
   cd LLM-Eval/
   pip install -e .                  # Install Default Dependencies
   # Additional options
   pip install -e '.[test]'
   pip install -e '.[all]'           
   ```


## Quick Start
   ```shell
   llm-eval -h
   # LLM serving performance evaluation
   llm-eval serving_perf_eval -h
   # Edge(AIPC) LLM performance evaluation for lenovo-demo server
   llm-eval lenovo_perf_eval -h
   ```

## Windows EXE Packaging

To package the application as a Windows executable:

1. Install PyInstaller:
```shell
pip install pyinstaller
```

2. Run the build script:
```shell
python build_windows_exe.py
```

This will generate a single executable file in the `dist` folder.

Note:
- The build script requires Python 3.10+
- Make sure all dependencies are installed before building
- The executable is built for Windows but can be created on macOS/Linux using Wine