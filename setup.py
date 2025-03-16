from setuptools import setup, find_packages

setup(
    name="flowchef",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "diffusers",
        "gradio==5.6.0", 
        "numpy",
        "Pillow",
        "torch",
        "torchvision",
        "transformers",
        "accelerate",
        "xformers",
        "sentencepiece",
    ],
)