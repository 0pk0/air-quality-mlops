from setuptools import setup, find_packages

setup(
    name="air-quality-mlops",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.6.0",
        "seaborn>=0.12.0",
        "scipy>=1.10.0",
        "joblib>=1.3.0",
        "requests>=2.31.0",
        "python-dotenv>=1.0.0"
    ]
)
