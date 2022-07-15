import os
from setuptools import setup, find_packages

PACKAGE_ROOT = os.path.dirname(os.path.realpath(__file__))
README_FILE = open(os.path.join(PACKAGE_ROOT, "README.md"), "r").read()

if __name__ == "__main__":
    setup(
        name="autoclip",
        version="0.2.0",
        description="Smart gradient clippers",
        long_description=README_FILE,
        long_description_content_type="text/markdown",
        url="https://github.com/HesitantlyHuman/autoclip",
        author="HesitantlyHuman",
        author_email="tannersims@hesitantlyhuman.com",
        license="MIT",
        packages=find_packages(),
    )
