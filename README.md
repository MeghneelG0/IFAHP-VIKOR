# IFAHP-VIKOR Requirements Prioritization

This project implements the Intuitionistic Fuzzy Analytic Hierarchy Process (IFAHP) combined with VIKOR method for requirements prioritization.

> **Note:** The Python scripts (including main.py) are currently under development. All main functionality is currently implemented in Jupyter notebooks as described below.

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/IFAHP-VIKOR.git
cd ifahpVikor_SRPM
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Jupyter Notebooks

The core functionality of this project is implemented in the following Jupyter notebooks:

- **ifahpCode_validation.ipynb**: Evaluation and validation script that reproduces the process described in the accompanying research paper. The Intuitionistic Fuzzy Preference Relation (IFPR) matrix used in this notebook is taken directly from the case study section of the research to demonstrate and verify the correctness of the implemented code.

- **ifahpVIKOR.ipynb**: Contains the overall implementation of the IFAHP-VIKOR hybrid model for general use.

- **ifahpvikorTCP.ipynb**: Applies the IFAHP-VIKOR approach to a large dataset with 2000 alternatives and 5 criteria for test case prioritization.

- **ifahpWeights_caseStudy.ipynb**: Provides methods for converting AHP weights to IFAHP weights for comparative analysis as described in the research paper.

- **mcdm.ipynb**: Contains implementations for comparing various hybrid MCDM models including AHP, FAHP, IFAHP with VIKOR and TOPSIS.

## Method Overview

This implementation combines two multi-criteria decision making approaches:

1. **IFAHP (Intuitionistic Fuzzy Analytic Hierarchy Process)**:
   - Generates weights for each criterion using intuitionistic fuzzy values
   - Ensures consistency in the preference matrix
   - Converts fuzzy weights to crisp values

2. **VIKOR (VIseKriterijumska Optimizacija I Kompromisno Resenje)**:
   - Uses the weights from IFAHP
   - Evaluates alternatives based on utility and regret measures
   - Identifies compromise solutions

## Future Development

The standalone Python scripts are under development and will eventually provide the same functionality as the Jupyter notebooks in a more modular format.
