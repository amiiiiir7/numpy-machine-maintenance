import numpy as np

np.random.seed(42)
np.set_printoptions(suppress=True, precision=2)

# Number of machines
N = 1000

# Generate machine IDs
machine_id = np.array([f"M{i:03d}" for i in range(1, N+1)])

# Generate baseline machine data
temperature = np.random.normal(60, 6, (N,))

vibration = np.random.exponential(1, (N,))+1
vibration = np.clip(vibration, 1, 10)

pressure = np.random.normal(12, 1, (N,))

running_hours = np.random.normal(1750, 500, (N,))
running_hours = np.clip(running_hours,0,5000)

energy_cons = np.random.exponential(100, (N,))+100
energy_cons = np.clip(energy_cons, 100, 1000)


# --------------------------------------------------
# Anomaly injection
# --------------------------------------------------

# Randomly select 70 unique machines
rand_indices = np.random.choice(N, 70, replace=False)

# Split selected machines into warning and critical groups
warning_indices = rand_indices[:50]
critical_indices = rand_indices[50:]

# Split warning machines into five groups of 10
warning_temp = warning_indices[:10]
warning_vibration = warning_indices[10:20]
warning_pressure = warning_indices[20:30]
warning_hours = warning_indices[30:40]
warning_energy = warning_indices[40:]

# Split critical machines into three groups
crit_temp = critical_indices[:7]
crit_vibration = critical_indices[7:14]
crit_pressure = critical_indices[14:]

# Inject warning-level anomalies
temperature[warning_temp] = np.random.randint(76, 85, size=10)
vibration[warning_vibration] = np.random.randint(5, 9, size=10)
pressure[warning_pressure] = np.random.randint(14, 16, size=10)
running_hours[warning_hours] = np.random.randint(4500, 5001, size=10)
energy_cons[warning_energy] = np.random.randint(750, 1001, size=10)

# Inject critical-level anomalies
temperature[crit_temp] = np.random.randint(85, 96, size=7)
vibration[crit_vibration] = np.random.randint(9,11, size=7)
pressure[crit_pressure] = np.random.randint(16, 18, size=6)

# -------------------------
# Build final dataset
# -------------------------

machines = np.stack(
    (temperature, vibration, pressure, running_hours, energy_cons),
    axis=1
)

# --------------------------------------------------
# Detect anomalies using thresholds
# --------------------------------------------------

# Temperature
temp_warning = (temperature >= 76) & (temperature <= 84)
temp_critical = temperature >= 85

temp_warning_indices = np.where(temp_warning)[0]
temp_critical_indices = np.where(temp_critical)[0]


# Vibration
vib_warning = (vibration >= 5) & (vibration <= 8)
vib_critical = vibration >= 9

vib_warning_indices = np.where(vib_warning)[0]
vib_critical_indices = np.where(vib_critical)[0]


# Pressure
pres_warning = (pressure >= 14) & (pressure <= 15)
pres_critical = pressure >= 16

pres_warning_indices = np.where(pres_warning)[0]
pres_critical_indices = np.where(pres_critical)[0]


# Running hours
# Warning indicator only
run_warning = running_hours >= 4500

run_warning_indices = np.where(run_warning)[0]


# Energy consumption
# Warning indicator only
energy_warning = energy_cons >= 750

energy_warning_indices = np.where(energy_warning)[0]


# --------------------------------------------------
# Detection summary
# --------------------------------------------------

print("Temperature:   Warning:", np.sum(temp_warning),
      " Critical:", np.sum(temp_critical))

print("Vibration:     Warning:", np.sum(vib_warning),
      " Critical:", np.sum(vib_critical))

print("Pressure:      Warning:", np.sum(pres_warning),
      " Critical:", np.sum(pres_critical))

print("Running hours: Warning:", np.sum(run_warning))

print("Energy:        Warning:", np.sum(energy_warning))


warning_mask = (temp_warning | vib_warning | pres_warning | run_warning | energy_warning)

critical_mask = (temp_critical | vib_critical | pres_critical)


status = np.full(N, 'Normal', dtype="<U8")

status[warning_mask] = 'Warning'

status[critical_mask] = 'Critical'


# ==================================================
#                 FINAL REPORT
# ==================================================

print("\n" + "=" * 70)
print("                      MACHINE MAINTENANCE REPORT")
print("=" * 70)


# --------------------------------------------------
# Overall Machine Status
# --------------------------------------------------

print("\nOverall Machine Status")
print("-" * 70)

normal_count = np.sum(status == "Normal")
warning_count = np.sum(status == "Warning")
critical_count = np.sum(status == "Critical")

print(f"Normal:    {normal_count:3d} machines ({normal_count / N * 100:.1f}%)")
print(f"Warning:   {warning_count:3d} machines ({warning_count / N * 100:.1f}%)")
print(f"Critical:  {critical_count:3d} machines ({critical_count / N * 100:.1f}%)")


# ==================================================
# Critical Machines
# ==================================================

critic_indices = np.where(critical_mask)[0]
critic_machines = machine_id[critic_indices]

print("\n" + "=" * 70)
print("                         CRITICAL MACHINES")
print("=" * 70)

print("\nAll Critical Machines:")
print(critic_machines)

print("\nCritical Machines due to Temperature:")
print(machine_id[temp_critical_indices])

print("\nCritical Machines due to Vibration:")
print(machine_id[vib_critical_indices])

print("\nCritical Machines due to Pressure:")
print(machine_id[pres_critical_indices])


# ==================================================
# Warning Machines
# ==================================================

warn_indices = np.where(warning_mask)[0]
warn_machines = machine_id[warn_indices]

print("\n" + "=" * 70)
print("                          WARNING MACHINES")
print("=" * 70)

print("\nAll Machines in Warning Situation:")
print(warn_machines)


# ==================================================
# End of Report
# ==================================================

print("\n" + "=" * 70)
print("                           END OF REPORT")
print("=" * 70)