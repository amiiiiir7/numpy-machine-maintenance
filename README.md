# NumPy Machine Maintenance

This is one of my learning projects while getting more comfortable with **Python and NumPy**.

I wanted to take a simple idea — monitoring a group of machines — and use it to practice NumPy in a way that feels a little closer to a real data problem.

> **Note:** This is a learning and portfolio project. The goal is to practice NumPy and understand the thinking behind the code, not to build a real predictive-maintenance system.

## The Idea

I simulate **1,000 machines** with five measurements:

- Temperature
- Vibration
- Pressure
- Running hours
- Energy consumption

The basic flow is:

**Generate → Add anomalies → Combine → Detect → Classify → Report**

## Data & Anomalies

The baseline data is generated using simple statistical distributions:

| Measurement | Method |
|---|---|
| Temperature | Normal (mean 60, std 6) |
| Vibration | Exponential + 1, clipped to 1–10 |
| Pressure | Normal (mean 12, std 1) |
| Running hours | Normal (mean 1750, std 500), clipped to 0–5000 |
| Energy consumption | Exponential + 100, clipped to 100–1000 |

I use a fixed random seed (`42`) so the results are reproducible.

To create something concrete to detect, I randomly select **70 unique machines** and inject controlled anomalies:

- 50 warning-level cases
- 20 critical-level cases

## Detection

I use simple threshold-based rules and NumPy Boolean masks.

**Warning:** temperature `76–84`, vibration `5–8`, pressure `14–15`, running hours `>= 4500`, energy `>= 750`.

**Critical:** temperature `>= 85`, vibration `>= 9`, pressure `>= 16`.

A machine is classified as:

- **Normal** — no abnormal condition
- **Warning** — at least one warning condition
- **Critical** — at least one critical condition

Critical takes priority if both conditions occur.

## What I Practiced

This project helped me connect NumPy concepts that I had previously studied separately:

- `np.random.normal()` and `np.random.exponential()`
- `np.random.choice()`
- `np.clip()`
- `np.stack()`
- Boolean masks and vectorized conditions
- `np.where()` for indices
- `np.sum()` for counting conditions
- NumPy advanced indexing

One of the useful lessons for me was understanding the difference between a **Boolean mask**, an **index**, and the **values selected by that index**.

## Result

The program finishes with a simple report showing machine status and machines that need attention.

With the current seed and settings:

- **89.2% Normal**
- **8.8% Warning**
- **2.0% Critical**

## Run the Project

You only need Python and NumPy:

```bash
python data-generator.py
```

## Learning Journey

I built this project mainly to make myself use NumPy rather than just read about it. It is intentionally simple; at this stage, I am more interested in understanding the fundamentals well and gradually taking on harder problems.

**Python → NumPy → Pandas → Machine Learning → AI**

This is one small step in that journey.