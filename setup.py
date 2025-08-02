import os
import re
# import socket
from setuptools import setup, find_packages
from setuptools.command.install import install


ROOT_DIR = os.path.dirname(__file__)


def get_path(*filepath) -> str:
    return os.path.join(ROOT_DIR, *filepath)


def find_version(filepath: str) -> str:
    """Extract version information from the given filepath.

    Adapted from https://github.com/ray-project/ray/blob/0b190ee1160eeca9796bc091e07eaebf4c85b511/python/setup.py
    """
    with open(filepath) as fp:
        version_match = re.search(
            r"^__version__ = ['\"]([^'\"]*)['\"]", fp.read(), re.M
        )
        if version_match:
            return version_match.group(1)
        raise RuntimeError("Unable to find version string.")


def get_version() -> str:
    return find_version(get_path('llm_eval', '__init__.py'))


def parse_requirements(fname='requirements.txt', with_version=True):
    """Parse the package dependencies listed in a requirements file but strips
    specific versioning information.

    Args:
        fname (str): path to requirements file
        with_version (bool, default=False): if True include version specs

    Returns:
        List[str]: list of requirements items

    CommandLine:
        python -c "import setup; print(setup.parse_requirements())"
    """
    import re
    import sys
    from os.path import exists
    require_fpath = fname

    def parse_line(line):
        """Parse information from a line in a requirements text file."""
        if line.startswith('-r '):
            # Allow specifying requirements in other files
            target = line.split(' ')[1]
            for info in parse_require_file(target):
                yield info
        else:
            info = {'line': line}
            if line.startswith('-e '):
                info['package'] = line.split('#egg=')[1]
            else:
                # Remove versioning from the package
                pat = '(' + '|'.join(['>=', '==', '>']) + ')'
                parts = re.split(pat, line, maxsplit=1)
                parts = [p.strip() for p in parts]

                info['package'] = parts[0]
                if len(parts) > 1:
                    op, rest = parts[1:]
                    if ';' in rest:
                        # Handle platform specific dependencies
                        # http://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-platform-specific-dependencies
                        version, platform_deps = map(str.strip,
                                                     rest.split(';'))
                        info['platform_deps'] = platform_deps
                    else:
                        version = rest  # NOQA
                    if '--' in version:
                        # the `extras_require` doesn't accept options.
                        version = version.split('--')[0].strip()
                    info['version'] = (op, version)
            yield info

    def parse_require_file(fpath):
        with open(fpath, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    for info in parse_line(line):
                        yield info

    def gen_packages_items():
        if exists(require_fpath):
            for info in parse_require_file(require_fpath):
                parts = [info['package']]
                if with_version and 'version' in info:
                    parts.extend(info['version'])
                if not sys.version.startswith('3.4'):
                    # apparently package_deps are broken in 3.4
                    platform_deps = info.get('platform_deps')
                    if platform_deps is not None:
                        parts.append(';' + platform_deps)
                item = ''.join(parts)
                yield item

    packages = list(gen_packages_items())
    return packages

def do_setup():
    
    install_requires = parse_requirements('requirements/runtime.txt')
    
    extra_requirements_files = {
        'nvidia': 'requirements/nvidia.txt',
        'amd': 'requirements/amd.txt',
        'enflame': 'requirements/enflame.txt',
        'test': 'requirements/test.txt',
    }

    # Directly merge all dependencies through the list
    extra_requires = {key: parse_requirements(file) for key, file in extra_requirements_files.items()}

    # Merge all dependencies
    all_requires = install_requires + sum(extra_requires.values(), [])
    extra_requires['all'] = all_requires
    
    setup(
        name='llm-eval',
        version=get_version(),
        author='LLM-Eval Team',
        author_email='howardchen0119@outlook.com',
        license='Apache 2.0',
        description='LLM-Eval: MxN Evaluation Framework',
        long_description=open('README.md').read(),
        long_description_content_type='text/markdown',
        url='https://github.com/HowardChenRV/LLM-Eval.git',
        packages=find_packages(exclude=('examples', 'tests*')),
        # cmdclass={
        #     'install': CustomInstall,
        # },
        install_requires=install_requires,
        classifiers=[
            'Development Status :: 4 - Beta',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3.10',
        ],
        python_requires='>=3.10',
        entry_points={
            'console_scripts': [
                'llm-eval=llm_eval.cli.main:run_command',
            ],
        },
        include_package_data=True,
        package_data={
            '': ['conf/config.yaml', 'conf/**/*.yaml'],
        },
        extras_require=extra_requires,
    )


if __name__ == '__main__':
    do_setup()
