# coding:utf-8

import re
import os
import shutil
from pathlib import Path
from setuptools import Extension, setup, find_packages
from setuptools.command.build_py import build_py as build_py_orig
from Cython.Distutils import build_ext
from Cython.Build import cythonize

pkg_name = 'asr_api_server'

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


#as specified by @hoefling to ignore .py and not resource_folder
class build_py(build_py_orig):
    def build_packages(self):
        for package in self.packages:
            if 'data_model' not in package:
                print('skip pacakage:', package)
                continue
            package_dir = self.get_package_dir(package)
            modules = self.find_package_modules(package, package_dir)
            # Now loop over the modules we found, "building" each one (just
            # copy it to self.build_lib).
            for (package_, module, module_file) in modules:
                assert package == package_
                self.build_module(module, module_file, package)


def main(use_cython=False):
    py_modules = [
        f'{pkg_name}/__init__',
        f'{pkg_name}/vad_api',
        f'{pkg_name}/config',
        f'{pkg_name}/logger',
        f'{pkg_name}/vad_processor',
        f'{pkg_name}/vad_gpvad',
        f'{pkg_name}/easytimer',
        f'{pkg_name}/downloader',
        f'{pkg_name}/vad_performance',
    ]
    package_data={
        f'{pkg_name}.gpvad': [
            'asr_api_server/gpvad/labelencoders/vad.pkl',
            'asr_api_server/gpvad/pretrained_models/t2bal/t2bal.pth',
        ],
        f'{pkg_name}.gpvad_onnx': [
            'asr_api_server/gpvad_onnx/labelencoders/vad.pkl',
            'asr_api_server/gpvad_onnx/onnx_models/t2bal.onnx',
        ]
    }
    pyfiles = [p + '.py' for p in py_modules]
    extensions = []
    for p in py_modules:
        ex = Extension(f'{p}', [f'{p}.py'])
        extensions.append(ex)
    extensions.extend([
        Extension(f'{pkg_name}/gpvad_onnx/*.so', [f"{pkg_name}/gpvad_onnx/*.py"]),
        Extension(f'{pkg_name}/gpvad/*.so', [f"{pkg_name}/gpvad/*.py"]),
    ])
    if use_cython:
        ext_modules = cythonize(
                extensions,
                compiler_directives=dict(always_allow_keywords=True),
                language_level=3)
        packages = [
            f'{pkg_name}.gpvad',
            f'{pkg_name}.gpvad_onnx',
            f'{pkg_name}.data_model',
        ]
        py_modules = []
    else:
        ext_modules = []
        packages = [
            f'{pkg_name}.gpvad',
            f'{pkg_name}.gpvad_onnx',
            f'{pkg_name}.data_model',
        ]
        
    setup(
        name=pkg_name,
        version=get_version(),
        description='语音识别服务',
        author='',
        author_email='',
        url='',
        packages=packages,
        py_modules=py_modules,
        ext_modules=ext_modules,
        entry_points={
            'console_scripts': [
                'vad_api_server=asr_api_server.vad_api:run_api',
            ]
        },
        include_package_data=True,
        exclude_package_data={"": ["*.c", "*.py"]},
        cmdclass={
            'build_py': build_py
        },


    )
    print(f'{packages = }')


if __name__ == '__main__':
    # set False if no need to cythonize
    use_cython = True 
    main(use_cython)
