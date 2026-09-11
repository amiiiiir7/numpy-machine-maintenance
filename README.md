# NumPy Machine Maintenance

This is one of my learning projects while I am getting more comfortable with **Python and NumPy**.

I wanted to take a simple idea — monitoring a group of machines — and use it as an opportunity to practice NumPy in a way that feels a little more like a real data problem.

The project generates synthetic machine measurements, adds some controlled anomalies, detects unusual values, and produces a simple maintenance report.

> **Note:** This is a learning and portfolio project. The goal is to practice NumPy and understand the thinking behind the code, not to build a real predictive-maintenance system.

## The Idea

I simulate **1,000 machines**. Each machine has five measurements:

- Temperature
- Vibration
- Pressure
- Running hours
- Energy consumption

The basic flow is:

**Generate → Add anomalies → Combine → Detect → Classify → Report**

I chose this structure because it gives me a small but complete data-analysis problem to work with.

## Generating the Data

For the baseline data, I used a few simple statistical assumptions:

| Measurement | Distribution / Method |
|---|---|
| Temperature | Normal distribution, mean 60, standard deviation 6 |
| Vibration | Exponential distribution + 1, clipped to 1–10 |
| Pressure | Normal distribution, mean 12, standard deviation 1 |
| Running hours | Normal distribution, mean 1750, standard deviation 500, clipped to 0–5000 |
| Energy consumption | Exponential distribution + 100, clipped to 100–1000 |

I also use a fixed random seed (`42`) so that I can get the same results when I run the program again.

## Adding Some Anomalies

The randomly generated data will naturally contain some high or unusual values, but I also wanted to create a controlled situation that I could test.

I randomly select **70 unique machines**:

- 50 receive warning-level anomalies
- 20 receive critical-level anomalies

The warning cases are spread across the five measurements. Critical cases are added to temperature, vibration, and pressure.

This gave me something concrete to detect instead of just generating data and looking at it.

## Detecting Anomalies

I use simple thresholds to decide whether a measurement is in a warning or critical range.

### Warning ranges

- Temperature: `76–84`
- Vibration: `5–8`
- Pressure: `14–15`
- Running hours: `>= 4500`
- Energy consumption: `>= 750`

### Critical ranges

- Temperature: `>= 85`
- Vibration: `>= 9`
- Pressure: `>= 16`

The detection itself is based on **Boolean masks**. For example, a warning mask combines the individual warning conditions:

```python
warning_mask = (
    temp_warning |
    vib_warning |
    pres_warning |
    run_warning |
    energy_warning
)
```

I found this part especially useful for understanding how NumPy can work with many values at once instead of checking each machine individually.

## From Conditions to Machine Status

After detecting the conditions, I classify each machine as:

- **Normal** — no warning or critical condition
- **Warning** — at least one warning condition and no critical condition
- **Critical** — at least one critical condition

If a machine has both warning and critical conditions, I treat it as **Critical**.

This is a simple rule-based approach, but it helped me understand how Boolean arrays can be combined to create a higher-level result.

## What I Learned

This project helped me connect several NumPy concepts that I had previously studied separately.

Some of the main things I practiced were:

- NumPy arrays and array shapes
- `np.random.normal()` and `np.random.exponential()`
- `np.random.choice()` for selecting unique indices
- `np.clip()` for controlling generated values
- `np.stack()` for building the final data array
- Boolean conditions and masks
- `np.where()` for getting indices
- `np.sum()` for counting `True` values
- NumPy advanced indexing
- Vectorized operations

More importantly, I started to get a better feeling for the difference between:

```text
mask        → tells me which positions satisfy a condition
np.where()  → gives me the positions
array[...]  → gives me or changes the values at those positions
np.sum()    → counts how many positions satisfy the condition
```

That was one of the useful lessons from this project.

## Final Report

The program finishes by printing a small report with the overall machine status and the machines that need attention.

With the current random seed and settings, the result is:

- **89.2% Normal**
- **8.8% Warning**
- **2.0% Critical**

The report also shows critical machines by measurement type and lists the machines in a warning situation.

## Project Structure

```text
numpy-machine-maintenance/
│
├── data-generator.py
└── README.md
```

## How to Run

You only need Python and NumPy.

```bash
python data-generator.py
```

## A Small Step in My Learning Journey

I built this project mainly to make myself use NumPy rather than just read about it.

It is not a sophisticated project, and that is intentional. At this stage, I am more interested in understanding the fundamentals well and gradually taking on harder problems.

This project is one step in the path I am following:

**Python → NumPy → Pandas → Machine Learning → AI**

I expect the next projects to become more data-oriented and eventually move from simple rules and analysis toward machine learning and prediction.

For now, this is simply one small project where I learned a little more than I knew before.
