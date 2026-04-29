from setuptools import setup, find_packages

setup(
    name="songsplat",
    version="0.1.0",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "songsplat=songsplat.cli.main:main",
        ],
    },
    python_requires=">=3.10",
)
