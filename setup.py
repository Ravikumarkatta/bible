from setuptools import setup, find_packages

setup(
    name='biblical_ai',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch',
        'transformers',
        'pandas',
        'beautifulsoup4'
    ],
    # Add this line to avoid the license files issue
    license_files=None
)