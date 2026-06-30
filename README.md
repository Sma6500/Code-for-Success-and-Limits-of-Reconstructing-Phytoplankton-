# Code-for-Success-and-Limits-of-Reconstructing-Phytoplankton-
Code for the article Success and Limits of Reconstructing Phytoplankton Pigment Temporal Variations  from Simultaneous Surface Ocean Physics with a Unet model

This repository contains the source code used to produce the results presented in the article:

> **Authors** (Year). *Article Title*. Journal. DOI: *doi-link*

## Overview

This repository provides the implementation used for the experiments, analyses, and figures presented in the accompanying publication. It is intended to facilitate reproducibility and serve as a reference for readers interested in the methodology.

## Repository Structure

```text
.
├── configs_runned/           # All config runned
├── notebooks     # Jupyter notebooks for analysis and visualization, review contains the plot for the review answer
├── main/Tester/Trainer/Model.py            # Source code, pytorch lightining like for training
├── utils/        # Helper scripts
├── models/        # Various architectures tested
└── README.md
```


## Data Availability

All data sources can be found in the Article or hereunder : 
\begin{enumerate}
       \item ACRI-ST (\href{ftp://ftp.hermes.acri.fr}{available here})\label{data:globc}
       \item PSC~\cite{el_hourany_phytoplankton_2019}, accessible on demand\label{data:psc}
       \item GLORYSV12~\cite{Glorysv12} (\href{https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/description}{accessible here})\label{data:Glorysv12}
       \item ERA5 (\href{https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview}{accessible here}), DOI: 10.24381/cds.adbb2d47\label{data:ERA5}
       \item OC-CCI~\cite{sathyendranath_ocean-colour_2019} (\href{https://www.oceancolour.org/}{available here})\label{data:occci}       \item ONI index (\href{https://www.climate.gov/news-features/understanding-climate/climate-variability-oceanic-nino-index}{available here})\label{data:ONI}
       \item World Ocean Atlas (\href{https://www.ncei.noaa.gov/products/world-ocean-atlas}{available here})\label{data:WOA}
\end{enumerate}

## Citation

If you use this code in your work, please cite the associated publication:

```bibtex
@article{yourcitation,
  author  = {...},
  title   = {...},
  journal = {...},
  year    = {...},
  doi     = {...}
}
```

## Contact

For questions, suggestions, or bug reports, please open an issue or contact the corresponding author.
