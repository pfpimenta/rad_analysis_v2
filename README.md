# rad_analysis with pandas for NSREC26

## Project Description
This repository contains a suite of Python scripts designed for the analysis of data from radiation experiments. The primary focus is on evaluating the reliability and performance of Machine Learning (ML) models, particularly those deployed on Tensor Processing Units (TPUs), under radiation exposure. The analysis integrates two main types of data:
1.  **ML Model Logs**: Detailed logs capturing the behavior and outputs of ML models during experimental runs.
2.  **Radiation Facility Data**: Information pertaining to the radiation environment, including beam characteristics, fluence, and experimental conditions.

The project aims to quantify the impact of radiation on ML models, identify Single Event Upsets (SDCs), and compute key metrics such as radiation cross-sections.

The V1 of the project can be found at https://github.com/brunoloureiro/rad_analysis/
It still 

This project provides tools to analyze radiation experiment data from CNAO and PARTREC (2026). It focuses on evaluating the reliability of Machine Learning models deployed on Edge TPUs (Coral) by detecting Silent Data Corruptions (SDCs) and computing radiation cross-sections.

## Installation

1. Create and activate a virtual environment:
```bash
   python3 -m venv venv_rad_analysis
   source venv_rad_analysis/bin/activate
```

2. Install dependencies.
```bash
    pip install -r rad_analysis/requirements.txt
```


## Usage

TODO remake this part !

Run the analysis for specific experiments or the combined NSREC26 results:

```bash
# Analyze individual experiments
python3 CNAO_2026_01_analysis.py 
python3 PARTREC_2026_01_analysis.py 

# Combined analysis
python3 NSREC26_analysis.py 
```

Run all analysis:
```bash
python3 2024_07_TRIUMF_analysis.py; python3 2026_03_08_ENFORSA_analysis.py; python3 CNAO_2026_01_analysis.py ;python3 PARTREC_2026_01_analysis.py ; python3 NSREC26_analysis.py
```


### Inputs
The scripts expect data in the `rad_analysis_v2/data/<experiment_name>/` directory:
- **Logs:** Experiment log files in `logs/<device-name>/`.
- **Run Info:** A CSV file named `<experiment_name>_runs_info.csv` containing timestamps and beam flux information.

## Output

Results are generated in `rad_analysis_v2/data/<experiment_name>/results/`:
- **CSVs:** 
    - `*_SDCs.csv`: Summary of detected SDC events. Each row correspond to 1 SDC.
    - `*_SDC_details.csv`: Details on corrupted elements per SDC. Each line correspond to a wrong output element, and thus *n* rows (where n >= 1) are associated with one SDC.
    - `*_cross_sections_per_model.csv`: Cross-section metrics aggregated by model.
- **Plots:** Plot images are saved in the `plots/` subdirectory.


## TODOs

* run_analysis.py script that can call each analysis just by changing a CLI parameter
* print visualization of the a single SDC in the format of a PNG image
* output a report.md (or PDF) with a human readable summary of the analysis
* output a analysis_summary.json, equivalent of report.md, but machine readable
* meta_analysis script that compares different experiments
* merge with https://github.com/brunoloureiro/rad_analysis/ , or at least incorporate the functionalities present in there that are not here yet