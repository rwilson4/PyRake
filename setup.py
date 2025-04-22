
from setuptools import setup, find_packages

setup(
    name="pyrake",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["numpy", "scipy", "matplotlib"],
    author="Your Name",
    description="Balancing weights optimization with bias-variance tradeoff and efficient frontier tracing",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License"
    ]
)
