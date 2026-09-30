from setuptools import setup, find_packages

setup(
    name="admm-playground",
    version="0.1.0",
    description="Clean ADMM implementations: LASSO, Robust PCA, Matrix Completion",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=["numpy", "scipy"],
    extras_require={"examples": ["matplotlib"], "test": ["pytest"]},
)
