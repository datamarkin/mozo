from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mozo",
    version="0.1.0",
    description="A Python library named mozo",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Emrah NAZIF",
    author_email="emrah@datamarkin.com",
    url="https://github.com/datamarkin/mozo",
    project_urls={
        "Bug Tracker": "https://github.com/datamarkin/mozo/issues",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    packages=find_packages(),
    python_requires=">=3.6",
    license="MIT",
)