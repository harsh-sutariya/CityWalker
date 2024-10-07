from setuptools import setup, find_packages

setup(
    name='your_project',
    version='0.0',
    packages=find_packages(),
    install_requires=[
        'pytorch-lightning',
        'wandb',
        'efficientnet_pytorch',
        'decord',
        'tqdm',
        'scipy',
        'numpy',
        'torchvision',
        'PyYAML',
        'matplotlib',
    ],
)
