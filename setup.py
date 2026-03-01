from setuptools import setup, find_packages

# Open README.md with UTF-8 encoding to avoid decoding issues
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="mlsynth",
    version="0.1.1",
    description="A Python package for advanced synthetic control methods.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Jared Greathouse",
    author_email="jgreathouse3@student.gsu.edu",
    url="https://github.com/jgreathouse9/mlsynth",
    packages=find_packages(include=["mlsynth", "mlsynth.*"]),
    include_package_data=True,
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "scikit-optimize>=0.9.0",
        "statsmodels>=0.14.0",
        "cvxpy>=1.4.0",   # ECOS + ECOS_BB included
        "ecos",
        "pydantic>=2.0.0",
        "screenot",
    ],
    classifiers=[  # Metadata about the package
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",  # Specify Python version compatibility
)
