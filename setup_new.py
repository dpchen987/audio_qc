# coding:utf-8

import re
from setuptools import Extension, setup
from setuptools.command.build_py import build_py
from Cython.Build import cythonize

pkg_name = 'asr_api_new'


def get_version():
    with open(f'{pkg_name}/__init__.py') as f:
        for line in f:
            m = re.findall(r'__version__.*=.*(\d+\.\d+\.\d+)', line)
            if not m:
                continue
            ver = m[0]
            return ver
    print('!!!! NO_VERSION')
    return 'NO_VERSION'


# as specified by @hoefling to ignore .py and not resource_folder for .so build
class build_py_skip(build_py):
    def build_packages(self):
        pass


def main(use_cython=False):
    extensions = [
        Extension(f'{pkg_name}/*.so', [f"{pkg_name}/*.py"]),
        Extension(f'{pkg_name}/gpvad_onnx/*.so', [f"{pkg_name}/gpvad_onnx/*.py"]),
    ]
    if use_cython:
        ext_modules = cythonize(
                extensions,
                compiler_directives=dict(always_allow_keywords=True),
                language_level=3)
        packages = [f'{pkg_name}.gpvad_onnx']
    else:
        ext_modules = []
        packages = [pkg_name, f'{pkg_name}.gpvad_onnx']
    package_data = {
        f'{pkg_name}.gpvad_onnx': [
            'labelencoders/vad.pkl',
            'labelencoders/haha.wav',
            'onnx_models/t2bal.onnx',
        ]
    }
    if use_cython:
        cmdclass = {'build_py': build_py_skip} 
    else:
        cmdclass = {}
    setup(
        name=pkg_name,
        version=get_version(),
        description='语音识别服务',
        author='',
        author_email='',
        url='',
        # packages = find_packages(),
        packages=packages,
        ext_modules=ext_modules,
        entry_points={
            'console_scripts': [
                f'asr_api_server={pkg_name}.webapi:run_api',
                f'asr_worker={pkg_name}.asr_worker:main',
            ]
        },
        package_data=package_data,
        # include_package_data=True,
        # exclude_package_data={"": ["*.c"]},
        cmdclass=cmdclass,
    )
    print(packages)
    print(package_data)
    print('done')


if __name__ == '__main__':
    # set False if no need to cythonize
    use_cython = False
    main(use_cython)
