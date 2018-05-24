Installation
============

Wikipedia2Vec can be installed from PyPI:

```bash
% pip install wikipedia2vec
```

Alternatively, you can install the development version of this software from the GitHub repository:

```bash
% git clone https://github.com/studio-ousia/wikipedia2vec.git
% cd wikipedia2vec
% pip install Cython
% ./cythonize.sh
% pip install .
```

Wikipedia2Vec requires the 64-bit version of Python, and can be run on Linux, Windows, and Mac OSX.
It currently depends on the following Python libraries: [click](http://click.pocoo.org/), [jieba](https://github.com/fxsjy/jieba), [joblib](https://pythonhosted.org/joblib/), [lmdb](https://lmdb.readthedocs.io/), [marisa-trie](http://marisa-trie.readthedocs.io/), [mwparserfromhell](https://mwparserfromhell.readthedocs.io/), [numpy](http://www.numpy.org/), [scipy](https://www.scipy.org/), [six](https://pythonhosted.org/six/), and [tqdm](https://github.com/tqdm/tqdm).

If you want to train embeddings on your machine, it is highly recommended to install a BLAS library.
We recommend using [OpenBLAS](https://www.openblas.net/) or [Intel Math Kernel Library](https://software.intel.com/en-us/mkl).
Note that, the BLAS library needs to be recognized properly from SciPy.
This can be confirmed by using the following command:

```bash
% python -c 'import scipy; scipy.show_config()'
```

To process Japanese Wikipedia dumps, it is also required to install [MeCab](http://taku910.github.io/mecab/) and [its Python binding](https://pypi.python.org/pypi/mecab-python3).
Furthermore, to use [ICU library](http://site.icu-project.org/) to split either words or sentences or both, you need to install the [C/C++ ICU library](http://site.icu-project.org/download) and the [PyICU](https://pypi.org/project/PyICU/) library.
