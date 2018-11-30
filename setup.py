from setuptools import setup, find_packages

setup(
    name="nanoparticle_optimization",
    version="0.1.0",
    long_description=__doc__,
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numba',
        'mbuild',
    ],
    zip_safe=False,
)
