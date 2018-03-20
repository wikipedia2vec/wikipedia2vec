# -*- coding: utf-8 -*-

import os
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext

try:
    import pypandoc
    long_description = pypandoc.convert('README.md', 'rst')
except(IOError, ImportError):
    long_description = open('README.md').read()


def list_c_files(package_dir='wikipedia2vec'):
    ret = []
    for (dir_name, _, files) in os.walk(package_dir):
        for file_name in files:
            (module_name, ext) = os.path.splitext(file_name)
            if ext == '.c':
                module_name = '.'.join(dir_name.split(os.sep) + [module_name])
                path = os.path.join(dir_name, file_name)
                ret.append((module_name, path))

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
    name='wikipedia2vec',
    version='0.1.7',
    description='A tool for learning vector representations of words and entities from Wikipedia',
    long_description=long_description,
    author='Studio Ousia',
    author_email='ikuya@ousia.jp',
    url='http://studio-ousia.github.io/wikipedia2vec/',
    packages=find_packages(exclude=('tests*',)),
    cmdclass=dict(build_ext=custom_build_ext),
    ext_modules=[Extension(module_name, [path]) for (module_name, path) in list_c_files()],
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'wikipedia2vec=wikipedia2vec.cli:cli',
        ]
    },
    keywords=['wikipedia', 'embedding', 'wikipedia2vec'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    install_requires=[
        'click',
        'gensim',
        'joblib',
        'marisa-trie',
        'mwparserfromhell',
        'numpy',
        'scipy',
        'six',
    ],
    setup_requires=['numpy'],
    tests_require=['nose'],
    test_suite='nose.collector',
)
