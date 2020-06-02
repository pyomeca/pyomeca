---
title: "`Pyomeca`: An Open-Source Framework for Biomechanical Analysis"
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
These data are typically multidimensional arrays structured around labels with arbitrary metadata \autoref{fig:biomech-data}.
Existing software solutions share some limitations.
Some of them are not free of charge [@Rasmussen2003-yv] or based on closed-source programming language [@Dixon2017-co; @Muller2019-vd].
Others do not leverage labels and metadata [@Walt2011-em; @Hachaj2019-tk; @Virtanen2020-zv].
Pyomeca is a python package designed to address these limitations.

![An example of biomechanical data with skin marker positions.
These data are inherently multidimensional and structured around labels.
Metadata are also needed to inform about important features of the experiment.\label{fig:biomech-data}](fig/biomech-data.pdf)

# References
