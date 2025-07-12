from setuptools import setup, find_packages

setup(
    name="hefesto_analysis",
    version="0.1.0",
    author="Donghao Zheng",
    description="A Python package for analyzing HeFESTo geophysical simulation output",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent"
    ],
    python_requires='>=3.10',
)
