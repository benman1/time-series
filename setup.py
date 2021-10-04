from setuptools import setup, find_packages


with open("requirements.txt") as f:
    install_requires = [
        req
        for req in
        f.read().strip().split("\n")
        if req
    ]

print(install_requires)

setup(
    name="time-series",
    version="0.2",
    description="Time-Series models with keras and Tensorflow",
    author="Ben Auffarth",
    url="https://github.com/benman1/time-series/tree/master",
    install_requires=install_requires,
    packages=find_packages(),
)
