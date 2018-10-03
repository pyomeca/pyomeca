from setuptools import setup

import versioneer

requirements = [
    'versioneer'
]

setup(
    name='pyomeca',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Pyomeca is a python library allowing to carry out a complete biomechanical analysis; in a simple, logical and concise way",
    author="Romain Martinez & Benjamin Michaud",
    author_email='martinez.staps@gmail.com',
    url='https://github.com/pyomeca/pyomeca',
    license='Apache 2.0',
    packages=['pyomeca'],
    install_requires=requirements,
    keywords='pyomeca',
    classifiers=[
        'Programming Language :: Python :: 3.6',
    ]
)
