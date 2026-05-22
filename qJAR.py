"""
Adaptive Coherence Under Interference: Quantum Simulation
==========================================================
Models two XY-coupled qubits as a systems analog of the Jamming
Avoidance Response (JAR).

Redwan Rahman


Repository: https://github.com/Red1-Rahman/adaptive-coherence-jar

Units
-----
We use units with ℏ = 1 throughout. Frequencies are expressed in GHz,
and time in ns. We treat frequencies as angular-frequency parameters
internally (ω = 2πf), ensuring ω·t is dimensionless.

Physical model
--------------
Two superconducting transmon qubits (A and B) are coupled via a
capacitive exchange (XY) interaction:

  H = (ω_A / 2) Z_A  +  (ω_B / 2) Z_B  +  J · (X_A⊗X_B + Y_A⊗Y_B)

where:
  ω_A = 2π f_A,  ω_B = 2π f_B
  J               (XY coupling strength, GHz)
  X, Y, Z         Pauli operators

The XY interaction is the standard effective model for capacitive
coupling in transmon architectures (Koch et al., PRA 2007). It enables
coherent excitation exchange between qubits. Effective exchange is
resonantly enhanced when ω_A ≈ ω_B due to energy matching in the
rotating frame, and becomes off-resonant when |ω_A − ω_B| >> J.

Qubit A is initialized in |+⟩, which is sensitive to phase disturbances.
After unitary evolution, we trace out qubit B and compute fidelity with
respect to |+⟩. This serves as an aggregate measure of coherence
preservation under crosstalk-induced unitary evolution.

Simulation method
-----------------
We use PennyLane's default.qubit (statevector simulator). Evolution is
implemented via first-order Trotterization:

  exp(-i H dt) ≈ exp(-i (ω_A/2) Z_A dt)
               · exp(-i (ω_B/2) Z_B dt)
               · exp(-i J X_A⊗X_B dt)
               · exp(-i J Y_A⊗Y_B dt)

No noise or decoherence model is included; this is an ideal closed-system
simulation capturing coherent crosstalk only.

Classical feedback control (JAR-inspired)
------------------------------------------
The adaptive rule acts on classical control parameters:

  if |f_A − f_B| < ε  →
      f_B ← f_B + δ_f · sign(f_B − f_A)

This represents a classical feedback layer acting on drive frequencies,
not a quantum operation. It is inspired by JAR-like adaptive detuning.

Usage
-----
  python jar_quantum_final.py
  python jar_quantum_final.py --white
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pennylane as qml

# ── CLI ─────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--white", action="store_true")
args = parser.parse_args()
DARK = not args.white

np.random.seed(42)

# ── Parameters ─────────────────────────────────────
f_A = 5.0
J = 0.15
t_evolve = 3.0

epsilon = 0.15
delta_f = 0.6

delta_range = np.linspace(-1.0, 1.0, 120)
J_range = np.linspace(0.01, 0.20, 30)

dev = qml.device("default.qubit", wires=2)

# ── XY evolution ───────────────────────────────────
def evolve_xy(omega_A, omega_B, J_val, t, n_steps=40):
    dt = t / n_steps
    for _ in range(n_steps):
        qml.RZ(omega_A * dt, wires=0)
        qml.RZ(omega_B * dt, wires=1)
        qml.IsingXX(2 * J_val * dt, wires=[0, 1])
        qml.IsingYY(2 * J_val * dt, wires=[0, 1])

# ── Classical control ──────────────────────────────
def jar_control(f_B_in, f_A_ref, eps, shift):
    if abs(f_A_ref - f_B_in) < eps:
        direction = np.sign(f_B_in - f_A_ref) if f_B_in != f_A_ref else 1.0
        return f_B_in + shift * direction
    return f_B_in

def static_detuning(f_B_in, shift=0.6, direction=1.0):
    return f_B_in + direction * shift

# ── QNodes ─────────────────────────────────────────
@qml.qnode(dev)
def fidelity_circuit(df_val, mode, static_sign=1.0):
    omega_A = 2 * np.pi * f_A
    f_B = f_A + df_val

    if mode == "jar":
        f_B = jar_control(f_B, f_A, epsilon, delta_f)
    elif mode == "static":
        f_B = static_detuning(f_B, delta_f, static_sign)

    omega_B = 2 * np.pi * f_B

    qml.Hadamard(wires=0)
    evolve_xy(omega_A, omega_B, J, t_evolve)
    return qml.density_matrix(wires=0)

def fidelity_plus(dm):
    plus = np.array([1.0, 1.0]) / np.sqrt(2)
    return float(np.real(plus @ dm @ plus))

# ── Sweep ──────────────────────────────────────────
fid_before, fid_after, fid_static = [], [], []

for df in delta_range:
    fid_before.append(fidelity_plus(fidelity_circuit(df, "none")))
    fid_after.append(fidelity_plus(fidelity_circuit(df, "jar")))
    fid_static_plus = fidelity_plus(fidelity_circuit(df, "static", 1.0))
    fid_static_minus = fidelity_plus(fidelity_circuit(df, "static", -1.0))
    fid_static.append(0.5 * (fid_static_plus + fid_static_minus))

fid_before = np.array(fid_before)
fid_after = np.array(fid_after)
fid_static = np.array(fid_static)

# ── Key metrics ────────────────────────────────────
jam = np.abs(delta_range) < epsilon
activation_rate = np.mean(jam)

f_min_b = fid_before[jam].min()
f_min_a = fid_after[jam].min()

f_mean_b = fid_before[jam].mean()
f_mean_a = fid_after[jam].mean()
f_mean_s = fid_static[jam].mean()

recovery = (f_mean_a - f_mean_b) / (1 - f_mean_b + 1e-9) * 100
jar_vs_static = f_mean_a - f_mean_s

demo_idx = np.argmin(np.abs(delta_range - 0.1))
demo_b = fid_before[demo_idx]
demo_a = fid_after[demo_idx]

print("Results")
print("--------")
print(f"JAR activation rate: {activation_rate:.3f}")
print(f"Min fidelity before: {f_min_b:.4f}")
print(f"Min fidelity after : {f_min_a:.4f}")
print(f"Mean before        : {f_mean_b:.4f}")
print(f"Mean static        : {f_mean_s:.4f}")
print(f"Mean after         : {f_mean_a:.4f}")
print(f"Recovery           : {recovery:.1f}%")
print(f"JAR - static mean  : {jar_vs_static:.4f}")
print(f"Demo 0.1 GHz       : {demo_b:.4f} → {demo_a:.4f}")