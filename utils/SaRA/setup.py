from setuptools import setup

setup(
    name="SaRA",
    version="0.1.0",
    author="Yiming Shi",
    packages=["minsara"],
    description="A PyTorch implementation of SaRA",
    license="MIT",
    install_requires=[
        "torch>=2.0.0",
    ],
)
