from setuptools import setup, find_packages

version = {}
with open("src/beliefmatching/_version.py") as fp:
    exec(fp.read(), version)

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="beliefmatching",
    version=version["__version__"],
    packages=find_packages("src"),
    url="https://github.com/oscarhiggott/BeliefMatching",
    description="A package for decoding quantum error correcting codes using belief-matching.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_dir={"": "src"},
    author="Oscar Higgott",
    install_requires=[
        "stim",
        "pymatching>=2.0.1",
        "ldpc",
        "sinter>=1.11",
        "numpy<=2.2.6",
    ],
)
