from pathlib import Path
from setuptools import setup, find_packages

ROOT = Path(__file__).parent
README = (ROOT / 'README.md').read_text()

setup(
    name='gon-pytorch',
    packages=find_packages(),
    version='0.1.1',
    license='MIT',
    description='Gradient Origin Networks for PyTorch',
    long_description=README,
    long_description_content_type='text/markdown',
    author='Kristian Klemon',
    author_email='kristian.klemon@gmail.com',
    url='https://github.com/kklemon/gon-pytorch',
    keywords=['artificial intelligence', 'deep learning'],
    install_requires=['torch']
)
