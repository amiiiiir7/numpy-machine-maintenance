import numpy as np

np.random.seed(42)
np.set_printoptions(suppress=True, precision=2)

# Number of machines
N = 1000

# Generate machine IDs
machine_id = [f"M{i:03d}" for i in range(1, N+1)]

# Generate baseline machine data
temperature = np.random.normal(60, 8, (N,))

vibration = np.random.exponential(1.5, (N,))+1
vibration = np.clip(vibration, 1, 10)

pressure = np.random.normal(12, 1, (N,))

running_hours = np.random.randint(0, 5001, (N,))

energy_cons = np.random.exponential(150, (N,))+100
energy_cons = np.clip(energy_cons, 100, 1000)

# Combine machine measurements into one 1000 × 5 array
machines = np.stack(
    (temperature, vibration, pressure, running_hours, energy_cons),
    axis=1
)

# --------------------------------------------------
# Anomaly injection
# --------------------------------------------------

# Randomly select 70 unique machines
rand_indices = np.random.choice(N, 70, False)

# Split selected machines into warning and critical groups
warning_indices = rand_indices[:50]
critical_indices = rand_indices[50:]

# Split warning machines into five groups of 10
warning_temp = warning_indices[:10]
warning_vibration = warning_indices[10:20]
warning_pressure = warning_indices[20:30]
warning_hours = warning_indices[30:40]
warning_energy = warning_indices[40:50]

# Split critical machines into three groups
crit_temp = critical_indices[:7]
crit_vibration = critical_indices[7:14]
crit_pressure = critical_indices[14:]

# Inject warning-level anomalies
temperature[warning_temp] = np.random.randint(76, 85, size=10)
vibration[warning_vibration] = np.random.randint(5, 9, size=10)
pressure[warning_pressure] = np.random.randint(14, 16, size=10)
running_hours[warning_hours] = np.random.randint(4250, 4751, size=10)
energy_cons[warning_energy] = np.random.randint(500, 751, size=10)

# Inject critical-level anomalies
temperature[crit_temp] = np.random.randint(85, 95, size=7)
vibration[crit_vibration] = np.random.randint(9,11,size=7)
pressure[crit_pressure] = np.random.randint(16, 18, size=6)


