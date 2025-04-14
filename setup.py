from setuptools import setup, find_packages

setup(
    name="microflow",
    version="0.1.0",
    description="A lightweight event-driven workflow system for building AI agents and processing pipelines",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Vikram Soni",
    author_email="vikram9880@gmail.com",
    url="https://github.com/vikramsoni2/microflow",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.9",
)
