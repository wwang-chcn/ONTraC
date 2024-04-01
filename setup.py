from setuptools import setup, find_packages


# Function to read the version from the package's version.py
def get_version():
    version = {}
    with open("ONTraC/version.py") as fp:
        exec(fp.read(), version)
    return version['__version__']


setup(
    name='ONTraC',
    version=get_version(),
    package_dir={'ONTraC': 'ONTraC'},  # Specify the root directory for package contents
    packages=['ONTraC', 'ONTraC.run', 'ONTraC.model', 'ONTraC.train', 'ONTraC.utils',
              'ONTraC.optparser'],  # Specify the root directory for finding packages
    scripts=['ONTraC/bin/createDataSet', 'ONTraC/bin/GP', 'ONTraC/bin/NTScore', 'ONTraC/bin/ONTraC'])
