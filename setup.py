from setuptools import setup, find_packages
from pip.req import parse_requirements
from pip.download import PipSession


setup(
    name="time-series",
    version="0.2",
    description="Time-Series models with keras and Tensorflow",
    author="Ben Auffarth",
    url="https://github.com/benman1/time-series/tree/master",
    requires=parse_requirements("requirements.txt", session=PipSession()),
    packages=find_packages(),
)
