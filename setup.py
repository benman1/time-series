from setuptools import setup, find_packages
from pip.req import parse_requirements
from pip.download import PipSession


setup(
    name="deepar",
    version="0.0.1",
    description="DeepAR tensorflow implementation",
    author="Alberto Arrigoni",
    author_email="arrigonialberto86@gmail.com",
    url="https://github.com/arrigonialberto86/deepar/tree/master",
    requires=parse_requirements("requirements.txt", session=PipSession()),
    packages=find_packages(),
)
