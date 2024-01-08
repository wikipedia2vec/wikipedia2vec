import os
import platform
import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


def list_cpp_files(package_dir="wikipedia2vec"):
    if sys.platform.startswith("win"):
        compile_args = []
        link_args = []
    elif platform.system() == "Darwin":
        compile_args = ["-Wno-unused-function", "-std=c++11", "-stdlib=libc++"]
        link_args = ["-std=c++11", "-stdlib=libc++"]
    else:
        compile_args = ["-Wno-unused-function", "-std=c++11"]
        link_args = ["-std=c++11"]

    ret = []
    for dir_name, _, files in os.walk(package_dir):
        for file_name in files:
            (module_name, ext) = os.path.splitext(file_name)
            if ext == ".cpp":
                module_name = ".".join(dir_name.split(os.sep) + [module_name])
                path = os.path.join(dir_name, file_name)
                ret.append(
                    (
                        module_name,
                        dict(
                            sources=[path], language="c++", extra_compile_args=compile_args, extra_link_args=link_args
                        ),
                    )
                )

    return ret


# Copied from https://github.com/RaRe-Technologies/gensim/blob/master/setup.py
class custom_build_ext(build_ext):
    def finalize_options(self):
        build_ext.finalize_options(self)

        # Prevent numpy from thinking it is still in its setup process:
        # https://docs.python.org/2/library/__builtin__.html#module-__builtin__
        if isinstance(__builtins__, dict):
            __builtins__["__NUMPY_SETUP__"] = False
        else:
            __builtins__.__NUMPY_SETUP__ = False

        import numpy

        self.include_dirs.append(numpy.get_include())


setup(
    cmdclass=dict(build_ext=custom_build_ext),
    ext_modules=[Extension(module_name, **kwargs) for (module_name, kwargs) in list_cpp_files()],
)
