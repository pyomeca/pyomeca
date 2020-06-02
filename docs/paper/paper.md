---
title: "`pyomeca`: An Open-Source Framework for Biomechanical Analysis"
tags:
  - python
  - biomechanics
  - electromyography
  - kinematics
  - dynamics
authors:
  - name: Romain Martinez
    orcid: 0000-0001-9681-9448
    affiliation: "1"
  - name: Benjamin Michaud
    orcid: 0000-0002-5031-1048
    affiliation: "1"
  - name: Mickael Begon
    orcid: 0000-0002-4107-9160
    affiliation: "1"
affiliations:
  - name: School of Kinesiology and Exercise Science, Faculty of Medicine, University of Montreal, Canada
    index: 1
date: 2 June 2020
bibliography: paper.bib
---

# Statement of Need

Biomechanics is defined as the study of the structure and function of biological systems by means of the methods of mechanics [@Hatze1974-zc].
While biomechanics branches into several subfields, the data used are remarkably similar.
The processing, analysis and visualization of these data could therefore be unified in a software package.
Most biomechanical data characterizing human and animal movement appear as temporal waveforms representing specific measures such as muscle activity or joint angles.
These data are typically multidimensional arrays structured around labels with arbitrary metadata (\autoref{fig:biomech-data}).
Existing software solutions share some limitations.
Some of them are not free of charge [@Rasmussen2003-yv] or based on closed-source programming language [@Dixon2017-co; @Muller2019-vd].
Others do not leverage labels and metadata [@Walt2011-em; @Hachaj2019-tk; @Virtanen2020-zv].
`pyomeca` is a python package designed to address these limitations.

![An example of biomechanical data with skin marker positions.
These data are inherently multidimensional and structured around labels.
Metadata are also needed to inform about important features of the experiment.\label{fig:biomech-data}](fig/biomech-data.pdf)

# Summary

As a python library, `pyomeca` enables extraction, processing and visualization of biomechanical data for use in research and education.
It is motivated by the need for simpler tools and more reproducible workflows allowing practitioners to focus on their specific interests and leaving `pyomeca` to handle the computational details for them.
`pyomeca` builds on the core scientific python packages, in particular `numpy` [@Walt2011-em], `scipy` [@Virtanen2020-zv], `matplotlib` [@Hunter2007-fv] and `xarray` [@Hoyer2017-sf].
By providing labeled querying and computation, efficient algorithms and persistent metadata, the integration of `xarray` facilitates usability, which is a step towards the adoption of programming in biomechanics.
`xarray` is designed as a general-purpose library and tries to avoid including domain specific functionalities --- but inevitably, the need for more domain specific logic arises.
`pyomeca` provides a biomechanics layer that supports specialized file formats (`c3d`, `mat`, `trc`, `sto`, `mot`, `csv` and `xlsx`) and implements signal processing and matrix manipulation routines commonly used in biomechanics.
`pyomeca` was written in a modular, object-oriented way, which makes it extensible and promotes the use of method chaining.
`pyomeca` follows software best practices by being fully tested, linted and type annotated --- ensuring that the package is easily distributable and modifiable.
In addition to the [static documentation and API reference](https://pyomeca.github.io/), `pyomeca` includes a set of Jupyter Notebooks with examples.
These notebooks can be read and executed by anyone with only a web browser through [binder](https://mybinder.org/).

# Features

`pyomeca` inherits from the `xarray` features set, which includes label-based indexing, arithmetic, aggregation and alignment, resampling and rolling window operations, plotting, missing data handling and out-of-core computation.
In addition, pyomeca has four data structures built upon `xarray`.
Each structure is associated with a specific biomechanical data type:

- `Angles`: joint angles,
- `Rototrans`: rototranslation matrix,
- `Analogs`: generic signals such as EMGs, force signals or any other analog signals,
- `Markers`: skin markers position.

While there are technically dozens of functions implemented in `pyomeca`, one can generally group them into two distinct categories: object creation and data processing.

## Object Creation

The starting point for working with `pyomeca` is to create an object with one of the specific methods associated with the different classes available.
`pyomeca` offers several ways to create these objects: by directly specifying the data, by sampling random data from distributions, by converting other data structures or by reading files (\autoref{fig:object-creation}).

![`pyomeca` offers several ways to create specialized data structures: from scratch (orange), from random data (red), from other data structures (blue) or from files (green).\label{fig:object-creation}](fig/object-creation.pdf)

## Data Processing

`pyomeca`'s main functionality is to offer dedicated biomechanical routines.
These features can be broadly grouped into different categories: filtering, signal processing, normalization, matrix manipulation and file output functions (\autoref{fig:data-processing}).

![`pyomeca` data processing capabilities are available through the `meca` `DataArrayAccessor` (e.g. `array.meca`) that allow to implement domain specific methods on `xarray` data objects.
These methods can be categorized into filters (orange), signal processing (red), normalization (blue), matrix manipulation (green) and file output (purple) routines.\label{fig:object-creation}](fig/object-creation.pdf)

# References
