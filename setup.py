import setuptools
from dig.version import __version__

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dig",
    version=__version__,
    author="***",
    author_email="***",
    # entry_points={
    #     'console_scripts': [
    #         'dig=dig.xxx.xxx'
    #     ]
    # },
    description="Dive into Graphs is a turnkey library for graph deep learning research.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/divelab/DIG",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    # install_requires=[''],
    python_requires='>=3.6',
)