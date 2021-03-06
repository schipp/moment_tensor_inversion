# Moment Tensor Inversion (DC-space grid search)

[![DOI](https://zenodo.org/badge/268167708.svg)](https://zenodo.org/badge/latestdoi/268167708)

Invert for the moment tensor of an (regional) earthquake by grid-searching the double-couple parameter space (strike, dip, rake) and comparing synthetic waveforms (computed) with recorded seismograms at each grid-point. Waveform-similarity is estimated by a combination of L1- and L2-norm. This code has been used in Schippkus et al. ([2019](http://doi.org/10.17738/ajes.2019.0010)) to determine the moment tensor of the ML 4.2 Alland 2016 main shock.

> Schippkus, S., Hausmann, H., Duputel, Z., Bokelmann, G., AlpArray Working Group. 2019. The Alland earthquake sequence in Eastern Austria: Shedding light on tectonic stress geometry in a key area of seismic hazard. Austrian J. Earth. Sci. 112(2), 182–194

## Requirements

- Python 3
- Relevant base Green's Functions for applicable event-station-distances (ZSS, ZDS, ZDD, RSS, RDS, RDD, TSS, TDS), e.g., computed by CPS (Herrmann, [2013](http://www.eas.slu.edu/eqc/eqccps.html)).

## TODO

- [ ] Move settings to external config file
- [ ] Document properly
- [ ] Add visualization scripts
