from setuptools import setup, find_packages

# Read README with UTF-8 encoding
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="mlsynth",
    version="0.1.2",
    description="A Python package for advanced synthetic control and experimental design methods.",
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

    extras_require={
        # Optional solver support
        "scip": [
            "pyscipopt>=4.4.0"
        ]
    },

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],

    python_requires=">=3.9",
)
