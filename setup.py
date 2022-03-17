from setuptools import setup, find_namespace_packages, find_packages

setup(
    name='python-dlshogi2',
    packages=find_packages(where='pydlshogi2'),
    package_dir={"": "pydlshogi2"}
)
