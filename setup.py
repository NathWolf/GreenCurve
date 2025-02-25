from setuptools import setup, find_packages

setup(
    name="GreenCurve",
    version="0.1.0",
    description="Predict the 24-hour renewable energy production curve for the upcoming day.",
    author="Nathalia Wolf",
    author_email="nathalia@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas",
        "numpy",
        "matplotlib",
        "prophet"  # or "fbprophet" if needed
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
