# IFAHP-VIKOR Requirements Prioritization

This project implements the Intuitionistic Fuzzy Analytic Hierarchy Process (IFAHP) combined with VIKOR method for requirements prioritization.

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/ifahpVikor_SRPM.git
cd ifahpVikor_SRPM
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the main script to process the requirements file and generate rankings:

```bash
python main.py
```

This will:
1. Load requirements from `Requirements.csv` (if available)
2. Perform IFAHP-VIKOR analysis
3. Display top ranked requirements
4. Save full rankings to `Ranked_Requirements.csv`

### Input Format

The `Requirements.csv` file should contain the following columns:
- `Req ID`: Requirement identifier
- `Req Name`: Short name of the requirement
- `Description`: Detailed description
- `Cost`: Cost rating (higher values = higher cost)
- `Value`: Value rating (higher values = more valuable)
- `Importance`: Importance level ('H', 'M', 'L')
- `Pre-requisite`: Any prerequisite requirements
- `Test Cases`: Test case information

Only Cost, Value, and Importance are used in the prioritization.

### Advanced Usage

To see detailed calculation steps and intermediate values, modify the `main()` function in `main.py` to set the verbose parameter to True:

```python
rankings, Q, S, R, compromise_solution, conditions_satisfied, data = compute_ifahp_vikor(verbose=True)
```

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

## Project Structure

- `main.py`: Main script for running the prioritization
- `ifahp_weights/`: Modules for IFAHP calculations
  - `pmc_matrix.py`: Perfect multiplicative consistent matrix construction
  - `consistency_check.py`: Consistency verification
  - `repair_matrix.py`: Matrix repair algorithms
  - `weights.py`: Weight calculation functions
- `vikor/`: Modules for VIKOR calculations
  - `decision_matrix.py`: Decision matrix operations
  - `utility_measures.py`: Utility and regret calculations
  - `ranking.py`: Ranking functions
  - `vikor_conditions.py`: Condition checking for compromise solutions
