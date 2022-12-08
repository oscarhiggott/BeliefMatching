from setuptools import setup, find_packages

setup(
    name='beliefmatching',
    packages=find_packages("src"),
    package_dir={'': 'src'},
    author='Oscar Higgott',
    install_requires=['stim', 'pymatching>=2.0.1', 'ldpc', 'sinter>=1.11.dev1670280005']
)
