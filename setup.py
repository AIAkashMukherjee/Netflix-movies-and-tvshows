from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = "-e ."

def get_requirements(file_path: str) -> List[str]:
    requirements = []
    with open(file_path) as file:
        requirements = file.readlines()
        requirements = [req.strip() for req in requirements]
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    return requirements

setup(
    name='netflix-movies-tv-shows',
    version='0.0.1',
    author='Akash Mukherjee',
    author_email='akashmukherjee0000@gmail.com',
    description='A package for analyzing Netflix movies and TV shows data',
    long_description='This package provides tools for analyzing Netflix movies and TV shows data.',
    long_description_content_type='text/markdown',
    install_requires=get_requirements('requirements.txt'),
    packages=find_packages(),
)

# python setup.py bdist_wheel
