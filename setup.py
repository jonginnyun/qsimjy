from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
import os
import sys
import subprocess


class CMakeBuildExt(build_ext):
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
        ]

        build_temp = os.path.abspath(self.build_temp)
        os.makedirs(build_temp, exist_ok=True)

        source_dir = os.path.abspath(os.path.dirname(ext.sources[0]))

        subprocess.check_call(["cmake", source_dir] + cmake_args, cwd=build_temp)
        subprocess.check_call(["cmake", "--build", "."], cwd=build_temp)


setup(
    name="qsimjy",
    version="0.1.1",
    description="A Python package that approximately calculates the electric and magnetic field calculated by a semiconductor quantum dot device with a cobalt micromagnet.",
    author="Jonginn Yun",
    author_email="alyuuv@snu.ac.kr/alyuuv@stanford.edu",
    packages=find_packages(),
    ext_modules=[
        Extension("_cxx_magcalc", ["src/c_magcalc_pybind.cpp"]),
        Extension("_cxx_potcalc", ["src/c_potcalc_pybind.cpp"]),
    ],
    cmdclass={"build_ext": CMakeBuildExt},
    python_requires=">=3.7",
    install_requires=[
        "numpy",  
        "ezdxf",
        "shapely",
        "ubermag"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    zip_safe=False,  # The package contains C extension, thereby zip_safe should be false.
)
