import setuptools
import pyCloudCompare.pyCloudCompare as cc

with open("Readme.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyCloudCompareCLI",
    version=cc.__version__,
    author=cc.__author__,
    author_email="",
    description=cc.__doc__,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dwoiwode/pyCloudCompareCLI",
    packages=setuptools.find_packages(),
    license=cc.__license__,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
