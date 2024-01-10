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

def get_package_data():
    from glob import glob
    files = []
    files += glob('asr_api_server/static/js/*.js')
    files += glob('asr_api_server/static/js/recorder/*.js')
    files += glob('asr_api_server/static/js/recorder/*/*.js')
    files += glob('asr_api_server/static/css/*.css')
    files += glob('asr_api_server/static/fonts/*')
    files += glob('asr_api_server/templates/*.html')
    files += glob('asr_api_server/gpvad_onnx/labelencoders/vad.pkl')
    files += glob('asr_api_server/gpvad_onnx/labelencoders/haha.wav')
    files += glob('asr_api_server/gpvad_onnx/onnx_models/*.onnx')
    files += glob('asr_api_server/gpvad/labelencoders/vad.pkl')
    files += glob('asr_api_server/gpvad/pretrained_models/*/*.pth')
    new = []
    for f in files:
        p = f.find('/')
        n = f[p+1:]
        new.append(n)
    return new

print(get_package_data())

class MyBuildExt(build_ext):
    def run(self):
        build_ext.run(self)
        build_dir = Path(self.build_lib)
        root_dir = Path(__file__).parent
        target_dir = build_dir if not self.inplace else root_dir
        self.copy_file(Path('main_folder') / '__init__.py', root_dir, target_dir)

    def copy_file(self, path, source_dir, destination_dir):
        if not (source_dir / path).exists():
            return
        shutil.copyfile(str(source_dir / path), str(destination_dir / path))

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
    # TODO: 编译没问题，但是运行时pydantic的数据模型报错，未来解决
    # 暂时不编译data_model,  需把该模块加到setup()的packages里面
    extensions = [
        Extension(f'{pkg_name}/*.so', [f"{pkg_name}/*.py"]),
        Extension(f'{pkg_name}/gpvad_onnx/*.so', [f"{pkg_name}/gpvad_onnx/*.py"]),
    ]
    if use_cython:
        ext_modules = cythonize(
                extensions,
                compiler_directives=dict(always_allow_keywords=True),
                language_level=3)
        # 解决 pydantic 的问题后，在ext_modules里面编译data_model，
        # 则 packages=[]
        packages = [f'{pkg_name}.gpvad_onnx', f'{pkg_name}.data_model',]  ## 防止把其它.py 文件也打包到.whl文件里面
    else:
        ext_modules = []
        packages = find_packages()
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
                'asr_api_server=asr_api_server.main:run',
            ]
        },
        package_data={f'{pkg_name}.gpvad_onnx': get_package_data()},
        include_package_data=True,
        exclude_package_data={"": ["*.c", "*.py"]},
    # cmdclass={
    #         'build_py': build_py
    #     },

    )
    print('sssss', packages)


if __name__ == '__main__':
    # set False if no need to cythonize
    use_cython = False
    main(use_cython)
