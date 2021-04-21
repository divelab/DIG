import setuptools
from dig.version import __version__

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dive_into_graphs",
    version=__version__,
    author="xxx",
    author_email="xxx",
    # entry_points={
    #     'console_scripts': [
    #         'dig=dig.xxx.xxx'
    #     ]
    # },
    maintainer="xxx",
    maintainer_email="xxx",
    license="xxx",
    keywords="xxx",
    description="DIG: Dive into Graphs is a turnkey library for graph deep learning research.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/divelab/DIG",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
    ],
    install_requires=['scipy',
                      'cilog',
                      'typed-argument-parser==1.5.4',
                      'captum==0.2.0',
                      'shap',
                      'IPython',
                      'tqdm'],
    python_requires='>=3.6',
)