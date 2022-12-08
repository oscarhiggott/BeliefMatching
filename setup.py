from setuptools import setup, find_packages

setup(
    name='beliefmatching',
    packages=find_packages("src"),
    package_dir={'': 'src'},
    author='Oscar Higgott',
    install_requires=['stim', 'pymatching']
)