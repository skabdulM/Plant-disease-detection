from setuptools import setup, find_packages

setup(
    name='plant-disease-detection',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'torchaudio',
        'pycocotools',
        'roboflow',
        'Flask',
        'ultralytics',
    ],
    entry_points={
        'console_scripts': [
            # Add your command line scripts here
        ],
    },
)
