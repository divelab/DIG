import setuptools
from dig.version import __version__

with open("README.md", "r") as fh:
    long_description = fh.read()

setup_requires = ['pytest-runner']
tests_require = ['pytest', 'pytest-cov', 'mock']

setuptools.setup(
    name="dive_into_graphs",
    version=__version__,
    author="DIVE Lab@TAMU",
    author_email="sji@tamu.edu",
    # entry_points={
    #     'console_scripts': [
    #         'dig=dig.xxx.xxx'
    #     ]
    # },
    maintainer="DIVE Lab@TAMU",
#     maintainer_email="xxx",
    license="GPLv3",
#     keywords="xxx",
    description="DIG: Dive into Graphs is a turnkey library for graph deep learning research.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/divelab/DIG",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    install_requires=['scipy',
                      'cilog',
                      'typed-argument-parser==1.5.4',
                      'captum==0.2.0',
                      'shap',
                      'IPython',
                      'tqdm',
                      'rdkit-pypi',
                      'pandas',
                      'sympy',
                      'pyscf==1.7.6.post1',
                      'hydra-core'],
    python_requires='>=3.6',
    setup_requires=setup_requires,
    tests_require=tests_require,
    extras_require={'test': tests_require},
    include_package_data=True
)
