# NumPy Machine Maintenance

A learning-focused NumPy project for generating synthetic machine measurements, injecting controlled anomalies, detecting abnormal conditions, and summarizing machine status.

> **Project scope:** This is a portfolio and learning project designed to demonstrate fundamental NumPy and data-analysis skills. It is not intended to represent a production predictive-maintenance system.

## Project Overview

The project simulates a dataset for **1,000 machines** using NumPy. Each machine has five measurements:

- Temperature
- Vibration
- Pressure
- Running hours
- Energy consumption

The workflow follows a simple data-analysis pipeline:

**Generate → Inject → Combine → Detect → Classify → Report**

## What This Project Demonstrates

- Creating and working with NumPy arrays
- Generating synthetic data with probability distributions
- Using `np.random.normal()` and `np.random.exponential()`
- Controlling reproducibility with `np.random.seed()`
- Limiting values with `np.clip()`
- Selecting unique observations with `np.random.choice()`
- Combining one-dimensional arrays with `np.stack()`
- Building Boolean masks from threshold conditions
- Extracting indices with `np.where()`
- Counting Boolean conditions with `np.sum()`
- Using NumPy advanced indexing to modify selected observations
- Applying vectorized logic to classify machine status

## Data Generation

The baseline measurements are generated using simple statistical assumptions:

| Measurement | Distribution / Method |
|---|---|
| Temperature | Normal distribution, mean 60, standard deviation 6 |
| Vibration | Exponential distribution + 1, clipped to 1–10 |
| Pressure | Normal distribution, mean 12, standard deviation 1 |
| Running hours | Normal distribution, mean 1750, standard deviation 500, clipped to 0–5000 |
| Energy consumption | Exponential distribution + 100, clipped to 100–1000 |

A fixed random seed (`42`) is used so the generated results are reproducible.

## Anomaly Injection

To create a controlled test scenario, **70 unique machines** are selected randomly:

- 50 machines receive warning-level anomalies.
- 20 machines receive critical-level anomalies.

Warning anomalies are distributed across all five measurements, while critical anomalies are introduced for temperature, vibration, and pressure.

This controlled injection makes it possible to test the detection logic rather than relying only on naturally generated extreme values.

## Detection Logic

Each measurement is evaluated using Boolean conditions and predefined thresholds.

### Warning indicators

- Temperature: `76–84`
- Vibration: `5–8`
- Pressure: `14–15`
- Running hours: `>= 4500`
- Energy consumption: `>= 750`

### Critical indicators

- Temperature: `>= 85`
- Vibration: `>= 9`
- Pressure: `>= 16`

The final machine status follows a simple rule:

- **Normal:** no warning or critical condition
- **Warning:** at least one warning condition and no critical condition
- **Critical:** at least one critical condition

Critical status overrides warning status when both occur for the same machine.

## NumPy Techniques in the Project

One of the main learning goals is understanding how NumPy represents and processes data.

For example:

```python
warning_mask = (
    temp_warning |
    vib_warning |
    pres_warning |
    run_warning |
    energy_warning
)
```

The Boolean mask identifies machines that satisfy at least one warning condition.

Then:

```python
warn_indices = np.where(warning_mask)[0]
```

extracts their array indices, while:

```python
warn_machines = machine_id[warn_indices]
```

uses NumPy indexing to retrieve the corresponding machine IDs.

## Final Report

The script produces a text-based maintenance report containing:

- Overall Normal / Warning / Critical counts
- Percentage of machines in each status
- List of all critical machines
- Critical machines by measurement type
- List of warning machines

With the fixed random seed and current configuration, the final classification contains approximately:

- **89.2% Normal**
- **8.8% Warning**
- **2.0% Critical**

## Project Structure

```text
numpy-machine-maintenance/
│
├── data-generator.py
└── README.md
```

## How to Run

Make sure Python and NumPy are installed, then run:

```bash
python data-generator.py
```

## Learning Objective

The purpose of this project is not to build a sophisticated maintenance model. Instead, it is a practical exercise in turning a simple analytical idea into NumPy code.

It represents one step in a broader learning path from **Python fundamentals → NumPy → Pandas → Machine Learning → AI**.

## Future Learning

Possible future projects will build on these foundations by introducing Pandas for structured data analysis and then machine-learning techniques for prediction and evaluation.
