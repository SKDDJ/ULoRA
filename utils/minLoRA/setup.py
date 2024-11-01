from setuptools import setup

setup(
    name="minLoRA",
    version="0.1.0",
    author="Yiming Shi",
    packages=["minlora"],
    description="A PyTorch re-implementation of variant LoRA",
    license="MIT",
    install_requires=[
        "torch>=2.0.0",
    ],
)
