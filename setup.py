from setuptools import setup, find_packages

# Read dependencies from requirements.txt
with open("requirements.txt") as f:
    requirements = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="fraud_detection",
    version="0.0.1",
    author="NullBitZer0",
    packages=find_packages(),
    install_requires=requirements,
)
