from setuptools import setup

# to install dependencies in a clean conda env, run: `conda env create -f env.yml`

# to install all dependencies (included dev) in a clean conda env, run: `conda env create -f env_dev.yml`

requirements = [
    'numpy',
    'pandas',
    'scipy',
    'vtk',
    'pyqt'
]

setup(
    name='pyomeca',
    version='2018.01.22',
    description="Toolbox for biomechanics analysis",
    author="Romain Martinez & Pariterre",
    author_email='',
    url='https://github.com/pyomeca/pyomeca',
    license='MIT license',
    packages=['pyomeca'],
    install_requires=requirements,
    keywords='pyomeca',
    classifiers=[
        'Programming Language :: Python :: 3.6',
    ]
)
