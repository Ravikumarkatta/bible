from setuptools import setup, find_packages

setup(
    name="biblical_ai",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        # Add other dependencies from requirements.txt
    ],
)