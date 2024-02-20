from setuptools import setup, find_packages

setup(
    name='ONTraC',
    version='1.1.2',
    package_dir={'ONTraC': 'ONTraC'},  # Specify the root directory for package contents
    packages=['ONTraC','ONTraC.run','ONTraC.model','ONTraC.train','ONTraC.utils','ONTraC.optparser'],  # Specify the root directory for finding packages
    # Add other package metadata (author, description, dependencies, etc.)
    # entry_points={
    #     'console_scripts': [
    #         'create-dataset = ONTraC.bin.createDataSet:main',
    #         'gp = ONTraC.bin.GP:main'
    #     ],
    # },
    scripts=['ONTraC/bin/createDataSet', 'ONTraC/bin/GP']
)