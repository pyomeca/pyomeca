from setuptools import setup

# to install dependencies in a clean conda env, run: `conda env create -f env.yml`

requirements = [
    'numpy',
    'pandas',
    'matplotlib',
    'vtk',
    'scipy',
    'pyqt',
    'boost',
    'rbdl',
    'dlib',
    'biorbd',
    'ezc3d'
]

setup(
    name='pyomeca',
    version='0.1.1',
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
