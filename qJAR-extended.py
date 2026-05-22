"""
Adaptive Coherence Under Interference: Quantum Simulation
==========================================================
Models two XY-coupled qubits as a systems analog of the Jamming
Avoidance Response (JAR).

Redwan Rahman


Repository: https://github.com/Red1-Rahman/adaptive-coherence-jar

Usage
-----
  python jar_quantum_final.py           # dark theme
  python jar_quantum_final.py --white   # white theme
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pennylane as qml

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--white", action="store_true",
                    help="White/print-friendly theme")
args = parser.parse_args()
DARK = not args.white

# ── Reproducibility ──────────────────────────────────────────────────────────
np.random.seed(42)

# ═══════════════════════════════════════════════════════════════════════════
#  PARAMETERS  (all in GHz / ns, with ℏ = 1)
# ═══════════════════════════════════════════════════════════════════════════
f_A       = 5.0    # qubit A drive frequency (GHz) — held fixed
J         = 0.15   # XY exchange coupling strength (GHz)
t_evolve  = 3.0    # evolution time (ns)

epsilon   = 0.15   # classical JAR threshold (GHz): trigger if |Δf| < ε
delta_f   = 0.6    # classical JAR shift magnitude (GHz)

# Δf sweep: f_B − f_A
delta_range = np.linspace(-1.0, 1.0, 120)   # GHz

# Coupling-strength sweep (panel b): fix Δf = 0.1 GHz, vary J
J_range = np.linspace(0.01, 0.20, 30)       # GHz
# ═══════════════════════════════════════════════════════════════════════════

# ── PennyLane device ─────────────────────────────────────────────────────────
# Wire 0 = qubit A  |  wire 1 = qubit B
dev = qml.device("default.qubit", wires=2)

# ── Trotterised XY evolution ──────────────────────────────────────────────────
def evolve_xy(omega_A, omega_B, J_val, t, n_steps=40):
    """
    First-order Trotter evolution under the XY Hamiltonian:

      H = (ω_A/2)Z_A + (ω_B/2)Z_B + J(X_A X_B + Y_A Y_B)

    Each Trotter step of duration dt = t/n_steps applies:
      RZ(ω_A · dt, wire=0)        ← free precession of qubit A
      RZ(ω_B · dt, wire=1)        ← free precession of qubit B
      IsingXX(2J · dt, [0,1])     ← XX exchange term
      IsingYY(2J · dt, [0,1])     ← YY exchange term

    IsingXX(θ) = exp(-i θ/2 · X⊗X),  IsingYY(θ) = exp(-i θ/2 · Y⊗Y).
    Together they implement exp(-i J dt (XX+YY)) to first order.
    """
    dt = t / n_steps
    for _ in range(n_steps):
        qml.RZ(omega_A * dt, wires=0)
        qml.RZ(omega_B * dt, wires=1)
        qml.IsingXX(2 * J_val * dt, wires=[0, 1])
        qml.IsingYY(2 * J_val * dt, wires=[0, 1])

# ── Classical JAR control rule ────────────────────────────────────────────────
def jar_control(f_B_in, f_A_ref, eps, shift):
    """
    Classical feedback rule: shift f_B away from the resonance window.
    This is a control policy, not a quantum gate.
    """
    if abs(f_A_ref - f_B_in) < eps:
        direction = np.sign(f_B_in - f_A_ref) if f_B_in != f_A_ref else 1.0
        return f_B_in + shift * direction
    return f_B_in

def static_detuning(f_B_in, shift=0.6, direction=1.0):
    return f_B_in + direction * shift

# ── QNodes ───────────────────────────────────────────────────────────────────
@qml.qnode(dev)
def fidelity_circuit(df_val, mode, static_sign=1.0):
    """
    Returns the reduced density matrix of qubit A after XY evolution.
    Qubit A: |+⟩ init.  Qubit B: |0⟩ init.
    """
    omega_A = 2 * np.pi * f_A
    f_B     = f_A + df_val
    if mode == "jar":
        f_B = jar_control(f_B, f_A, epsilon, delta_f)
    elif mode == "static":
        f_B = static_detuning(f_B, delta_f, static_sign)
    omega_B = 2 * np.pi * f_B

    qml.Hadamard(wires=0)                          # qubit A → |+⟩
    evolve_xy(omega_A, omega_B, J, t_evolve)       # coupled evolution
    return qml.density_matrix(wires=0)             # trace out qubit B

@qml.qnode(dev)
def bloch_circuit(df_val, apply_jar):
    """Returns Pauli expectation values ⟨X⟩, ⟨Y⟩, ⟨Z⟩ for qubit A."""
    omega_A = 2 * np.pi * f_A
    f_B     = f_A + df_val
    if apply_jar:
        f_B = jar_control(f_B, f_A, epsilon, delta_f)
    omega_B = 2 * np.pi * f_B

    qml.Hadamard(wires=0)
    evolve_xy(omega_A, omega_B, J, t_evolve)
    return [qml.expval(qml.PauliX(0)),
            qml.expval(qml.PauliY(0)),
            qml.expval(qml.PauliZ(0))]

@qml.qnode(dev)
def fidelity_vs_J_circuit(J_val, apply_jar):
    """Fidelity at fixed Δf = 0.1 GHz, varying J."""
    omega_A = 2 * np.pi * f_A
    f_B     = f_A + 0.1
    if apply_jar:
        f_B = jar_control(f_B, f_A, epsilon, delta_f)
    omega_B = 2 * np.pi * f_B

    qml.Hadamard(wires=0)
    evolve_xy(omega_A, omega_B, J_val, t_evolve)
    return qml.density_matrix(wires=0)

# ── Fidelity metric ───────────────────────────────────────────────────────────
def fidelity_plus(dm):
    """
    F(ρ_A, |+⟩) = ⟨+|ρ_A|+⟩

    Measures how close qubit A's reduced state is to the ideal |+⟩ target.
    F = 1: no crosstalk disturbance. F < 1: XY coupling has mixed the state.
    """
    plus = np.array([1.0, 1.0]) / np.sqrt(2)
    return float(np.real(plus @ dm @ plus))

# ── Main sweep ────────────────────────────────────────────────────────────────
print("=" * 62)
print("  JAR Quantum Simulation — XY-coupled transmon model")
print("=" * 62)
print(f"  ℏ = 1  |  f_A = {f_A} GHz  |  J = {J} GHz  |  t = {t_evolve} ns")
print(f"  Δf sweep: {delta_range[0]:.1f} → {delta_range[-1]:.1f} GHz  ({len(delta_range)} points)")
print()

fid_before = []
fid_after  = []
fid_static = []

for i, df in enumerate(delta_range):
    fid_before.append(fidelity_plus(fidelity_circuit(df, "none")))
    fid_after.append( fidelity_plus(fidelity_circuit(df, "jar")))
    fid_static_plus = fidelity_plus(fidelity_circuit(df, "static", 1.0))
    fid_static_minus = fidelity_plus(fidelity_circuit(df, "static", -1.0))
    fid_static.append(0.5 * (fid_static_plus + fid_static_minus))
    if (i + 1) % 20 == 0:
        print(f"  [{i+1:3d}/{len(delta_range)}] Δf = {df:+.2f} GHz | "
              f"F_before = {fid_before[-1]:.4f} | "
              f"F_static = {fid_static[-1]:.4f} | "
              f"F_after = {fid_after[-1]:.4f}")

fid_before = np.array(fid_before)
fid_after  = np.array(fid_after)
fid_static = np.array(fid_static)

# ── Coupling-strength sweep ───────────────────────────────────────────────────
print("\n  Coupling-strength sweep (Δf = 0.10 GHz fixed)...")
fid_J_before, fid_J_after = [], []
for J_val in J_range:
    fid_J_before.append(fidelity_plus(fidelity_vs_J_circuit(J_val, apply_jar=False)))
    fid_J_after.append( fidelity_plus(fidelity_vs_J_circuit(J_val, apply_jar=True)))
fid_J_before = np.array(fid_J_before)
fid_J_after  = np.array(fid_J_after)

# ── Bloch vectors at demo point Δf = 0.1 GHz ─────────────────────────────────
bloch_b = np.array(bloch_circuit(0.1, apply_jar=False), dtype=float)
bloch_a = np.array(bloch_circuit(0.1, apply_jar=True),  dtype=float)
purity_b = np.linalg.norm(bloch_b)
purity_a = np.linalg.norm(bloch_a)

# ── Key scalars ───────────────────────────────────────────────────────────────
jam_mask     = np.abs(delta_range) < epsilon
activation_rate = np.mean(jam_mask)
f_min_b      = fid_before[jam_mask].min()
f_min_a      = fid_after[jam_mask].min()
f_min_s      = fid_static[jam_mask].min()
f_mean_b     = fid_before[jam_mask].mean()
f_mean_a     = fid_after[jam_mask].mean()
f_mean_s     = fid_static[jam_mask].mean()
recovery_pct = (f_mean_a - f_mean_b) / (1.0 - f_mean_b + 1e-9) * 100
jar_vs_static = f_mean_a - f_mean_s

demo_idx = np.argmin(np.abs(delta_range - 0.1))
demo_b   = fid_before[demo_idx]
demo_a   = fid_after[demo_idx]

print()
print("=" * 62)
print("  Results")
print("=" * 62)
print(f"  Crosstalk well  |Δf| < ε = {epsilon} GHz")
print(f"  JAR activation rate          : {activation_rate:.3f}")
print(f"  Min fidelity   before JAR control : {f_min_b:.4f}")
print(f"  Min fidelity   static detuning    : {f_min_s:.4f}")
print(f"  Min fidelity   after  JAR control : {f_min_a:.4f}")
print(f"  Mean fidelity  before JAR control : {f_mean_b:.4f}")
print(f"  Mean fidelity  static detuning    : {f_mean_s:.4f}")
print(f"  Mean fidelity  after  JAR control : {f_mean_a:.4f}")
print(f"  Coherence recovery               : +{recovery_pct:.1f}% of lost fidelity")
print(f"  Mean gain (JAR - static)         : {jar_vs_static:.4f}")
print()
print(f"  Demo point Δf = 0.10 GHz")
print(f"    Fidelity   : {demo_b:.4f} → {demo_a:.4f}")
print(f"    Bloch |r|  : {purity_b:.4f} → {purity_a:.4f}  (1.0 = pure state)")
print(f"    ⟨X,Y,Z⟩ before : [{bloch_b[0]:.3f}, {bloch_b[1]:.3f}, {bloch_b[2]:.3f}]")
print(f"    ⟨X,Y,Z⟩ after  : [{bloch_a[0]:.3f}, {bloch_a[1]:.3f}, {bloch_a[2]:.3f}]")
print("=" * 62)

# ── Theme ─────────────────────────────────────────────────────────────────────
if DARK:
    FIG_BG  = '#0d1117'; AX_BG = '#161b22'; TEXT    = '#e6edf3'
    SUBTEXT = '#8b949e'; GRID  = '#21262d'; SPINE   = '#30363d'
    LEGEND  = '#21262d'
else:
    FIG_BG  = '#ffffff'; AX_BG = '#f6f8fa'; TEXT    = '#1f2328'
    SUBTEXT = '#57606a'; GRID  = '#d0d7de'; SPINE   = '#d0d7de'
    LEGEND  = '#ffffff'

COL_B    = '#f78166' if DARK else '#cf222e'   # before JAR
COL_A    = '#3fb950' if DARK else '#1a7f37'   # after JAR
COL_JAM  = '#f78166'
COL_IDEAL= '#58a6ff' if DARK else '#0969da'

def style_ax(ax, title, xlabel, ylabel):
    ax.set_facecolor(AX_BG)
    ax.set_title(title, color=TEXT, fontsize=9.5, fontweight='semibold', pad=6)
    ax.set_xlabel(xlabel, color=SUBTEXT, fontsize=8.5)
    ax.set_ylabel(ylabel, color=SUBTEXT, fontsize=8.5)
    ax.tick_params(colors=SUBTEXT, labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor(SPINE)
    ax.grid(True, color=GRID, linewidth=0.6, linestyle='--', alpha=0.8)

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 11), facecolor=FIG_BG)
fig.suptitle(
    "Adaptive Coherence Under Interference — Quantum Simulation\n"
    "XY-Coupled Transmon Crosstalk and JAR-Inspired Classical Feedback Control  "
    f"(J = {J} GHz,  t = {t_evolve} ns,  ℏ = 1)",
    fontsize=12, color=TEXT, fontweight='bold', y=0.99
)
gs = gridspec.GridSpec(2, 3, figure=fig,
                       hspace=0.50, wspace=0.36,
                       top=0.91, bottom=0.08,
                       left=0.08, right=0.97)

# ── (a) Fidelity vs Δf ───────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :2], facecolor=AX_BG)
ax1.axvspan(-epsilon, epsilon, alpha=0.10, color=COL_JAM,
            label=f'Crosstalk well  |Δf| < {epsilon} GHz')
ax1.plot(delta_range, fid_before, color=COL_B, lw=2.0, zorder=3,
         label='Without classical feedback')
ax1.plot(delta_range, fid_after,  color=COL_A, lw=2.0, zorder=3,
         ls='--', label='With JAR-inspired feedback')
ax1.plot(delta_range, fid_static, color=SUBTEXT, lw=1.6, zorder=2,
         ls=':', label='Static detuning baseline (+/- delta_f avg)')
ax1.axhline(1.0, color=COL_IDEAL, lw=0.8, ls=':', alpha=0.5,
            label='Ideal  F = 1')
ax1.annotate(f'F: {demo_b:.3f} → {demo_a:.3f}',
             xy=(0.1, demo_b), xytext=(0.38, demo_b - 0.045),
             arrowprops=dict(arrowstyle='->', color=TEXT, lw=1.0),
             fontsize=8, color=TEXT)
ax1.set_xlim(delta_range[0], delta_range[-1])
ax1.set_ylim(0.55, 1.06)
style_ax(ax1,
         '(a)  Qubit A fidelity F(ρ_A, |+⟩) vs. detuning  Δf = f_B − f_A',
         'Frequency detuning  Δf  (GHz)',
         'State fidelity  F(ρ_A, |+⟩)')
ax1.legend(fontsize=8, framealpha=0.4, labelcolor=TEXT,
           facecolor=LEGEND, edgecolor=SPINE, loc='lower right')

# ── (b) Fidelity vs J ────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 2], facecolor=AX_BG)
ax2.plot(J_range, fid_J_before, color=COL_B, lw=2.0,
         label='Without feedback')
ax2.plot(J_range, fid_J_after,  color=COL_A, lw=2.0, ls='--',
         label='With feedback')
ax2.axvline(J, color=SUBTEXT, lw=0.9, ls=':', alpha=0.8)
ax2.text(J + 0.004, 0.98, f'J = {J} GHz\n(used in sweep)',
         color=SUBTEXT, fontsize=7, va='top')
style_ax(ax2,
         f'(b)  Fidelity vs. coupling J\n(Δf = 0.10 GHz fixed)',
         'XY coupling  J  (GHz)',
         'State fidelity  F')
ax2.legend(fontsize=8, framealpha=0.4, labelcolor=TEXT,
           facecolor=LEGEND, edgecolor=SPINE)
ax2.set_ylim(0.45, 1.06)

# ── (c) Improvement map ──────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0], facecolor=AX_BG)
delta_F = fid_after - fid_before
ax3.fill_between(delta_range, delta_F, 0,
                 where=delta_F > 0,
                 color=COL_A, alpha=0.50, label='Fidelity gain from feedback')
ax3.fill_between(delta_range, delta_F, 0,
                 where=delta_F <= 0,
                 color=COL_B, alpha=0.25, label='No gain (already detuned)')
ax3.plot(delta_range, delta_F, color=COL_A, lw=1.2)
ax3.axhline(0, color=SPINE, lw=0.8)
ax3.axvspan(-epsilon, epsilon, alpha=0.07, color=COL_JAM)
style_ax(ax3,
         '(c)  Fidelity improvement  ΔF = F_after − F_before',
         'Frequency detuning  Δf  (GHz)',
         'ΔF (improvement)')
ax3.legend(fontsize=7.5, framealpha=0.4, labelcolor=TEXT,
           facecolor=LEGEND, edgecolor=SPINE)

# ── (d) Bloch vector ─────────────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1], facecolor=AX_BG)
comp  = ['⟨X⟩', '⟨Y⟩', '⟨Z⟩']
xp    = np.array([0.0, 1.0, 2.0])
w     = 0.30

bars_b = ax4.bar(xp - w/2, bloch_b, w, color=COL_B, alpha=0.82,
                 label='Without feedback', edgecolor=SPINE, linewidth=0.8)
bars_a = ax4.bar(xp + w/2, bloch_a, w, color=COL_A, alpha=0.82,
                 label='With feedback',    edgecolor=SPINE, linewidth=0.8)

for bar, val in list(zip(bars_b, bloch_b)) + list(zip(bars_a, bloch_a)):
    offset = 0.014 if val >= 0 else -0.028
    ax4.text(bar.get_x() + bar.get_width() / 2, val + offset,
             f'{val:.3f}', ha='center', va='bottom', color=TEXT, fontsize=7.5)

ax4.set_xticks(xp)
ax4.set_xticklabels(comp, color=TEXT, fontsize=10)
ax4.axhline(0, color=SPINE, lw=0.8)
ax4.set_ylim(-0.25, 1.15)
ax4.text(0.97, 0.97,
         f'Bloch vector  |r|\n'
         f'without: {purity_b:.4f}\n'
         f'with:      {purity_a:.4f}\n'
         f'(1.0 = pure state)',
         transform=ax4.transAxes, ha='right', va='top',
         color=TEXT, fontsize=7.5,
         bbox=dict(boxstyle='round,pad=0.35', facecolor=AX_BG,
                   edgecolor=SPINE, alpha=0.85))
style_ax(ax4,
         '(d)  Bloch vector  ⟨X⟩, ⟨Y⟩, ⟨Z⟩  at Δf = 0.10 GHz',
         'Pauli observable',
         'Expectation value')
ax4.legend(fontsize=8, framealpha=0.4, labelcolor=TEXT,
           facecolor=LEGEND, edgecolor=SPINE)

# ── (e) Model schematic ───────────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 2], facecolor=AX_BG)
ax5.set_xlim(0, 1); ax5.set_ylim(0, 1); ax5.axis('off')
ax5.set_title('(e)  Model summary',
              color=TEXT, fontsize=9.5, fontweight='semibold', pad=6)

# Qubit circles
ax5.add_patch(plt.Circle((0.22, 0.80), 0.09, color=COL_IDEAL, alpha=0.20, lw=0))
ax5.add_patch(plt.Circle((0.22, 0.80), 0.09, fill=False, color=COL_IDEAL, lw=1.5))
ax5.text(0.22, 0.80, 'A', ha='center', va='center',
         color=TEXT, fontsize=12, fontweight='bold')
ax5.text(0.22, 0.66, f'f_A = {f_A} GHz\n|+⟩', ha='center',
         color=SUBTEXT, fontsize=7.5)

ax5.add_patch(plt.Circle((0.78, 0.80), 0.09, color=COL_B, alpha=0.18, lw=0))
ax5.add_patch(plt.Circle((0.78, 0.80), 0.09, fill=False, color=COL_B, lw=1.5))
ax5.text(0.78, 0.80, 'B', ha='center', va='center',
         color=TEXT, fontsize=12, fontweight='bold')
ax5.text(0.78, 0.66, f'f_B = f_A + Δf\n|0⟩', ha='center',
         color=SUBTEXT, fontsize=7.5)

# XY coupling arrow + label
ax5.annotate('', xy=(0.67, 0.80), xytext=(0.33, 0.80),
             arrowprops=dict(arrowstyle='<->', color=COL_JAM, lw=1.8))
ax5.text(0.50, 0.86, 'J·(XX+YY)', ha='center',
         color=COL_JAM, fontsize=8, fontweight='bold')
ax5.text(0.50, 0.91, 'XY exchange coupling', ha='center',
         color=SUBTEXT, fontsize=7)

# Hamiltonian box — now correctly showing XY
ax5.text(0.50, 0.54,
         'H = (ω_A/2)Z_A + (ω_B/2)Z_B\n    + J(X_A X_B + Y_A Y_B)',
         ha='center', va='center', color=TEXT, fontsize=8,
         bbox=dict(boxstyle='round,pad=0.4', facecolor=AX_BG,
                   edgecolor=SPINE, alpha=0.9))

# JAR rule box — explicitly labelled as classical feedback
ax5.text(0.50, 0.34,
         'Classical feedback rule (JAR analog):\n'
         f'if  |f_A − f_B| < ε = {epsilon} GHz\n'
         f'then  f_B ← f_B + {delta_f}·sign(f_B − f_A) GHz',
         ha='center', va='center', color=COL_A, fontsize=7.5,
         bbox=dict(boxstyle='round,pad=0.4', facecolor=AX_BG,
                   edgecolor=COL_A, alpha=0.85, lw=1.2))

# Result summary
ax5.text(0.50, 0.10,
         f'F: {demo_b:.3f} → {demo_a:.3f}   '
         f'|r|: {purity_b:.3f} → {purity_a:.3f}  (Δf = 0.10 GHz)',
         ha='center', color=TEXT, fontsize=7.5, fontweight='bold')

# ── Save ──────────────────────────────────────────────────────────────────────
theme    = 'white' if not DARK else 'dark'
out_name = f'jar_quantum_{theme}.png'
out_path = out_name  # Save to current working directory
plt.savefig(out_path, dpi=180, bbox_inches='tight', facecolor=FIG_BG)
print(f"\n✓ Figure saved → {out_path}")
plt.close()