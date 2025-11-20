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
    install_requires=[  # Dependencies from requirements.txt
        "pandas",
        "numpy",
        "matplotlib",
        "scipy",
        "scikit-learn",
        "statsmodels",
        "cvxpy",
        "pydantic"
    ],
    classifiers=[  # Metadata about the package
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",  # Specify Python version compatibility
)
