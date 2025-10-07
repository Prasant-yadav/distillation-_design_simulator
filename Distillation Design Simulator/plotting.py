import matplotlib.pyplot as plt
import numpy as np


def create_mccabe_thiele_plot(xF, xD, xB, R, alpha, q):
    """
    Creates a complete McCabe-Thiele diagram for a binary distillation system.

    Args:
        xF (float): Mole fraction of the light component in the feed.
        xD (float): Mole fraction of the light component in the distillate.
        xB (float): Mole fraction of the light component in the bottoms.
        R (float):  Operating reflux ratio.
        alpha (float): Relative volatility of the light component to the heavy.
        q (float): Quality of the feed (e.g., 1 for sat. liquid, 0 for sat. vapor).

    Returns:
        tuple: A tuple containing the plot figure, actual stages, feed stage, and R_min.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # --- 1. Calculate Line Parameters ---
    slope_rect = R / (R + 1)
    intercept_rect = xD / (R + 1)

    if abs(q - 1.0) < 1e-6:
        slope_q = float('inf')
        x_int = xF
        y_int = slope_rect * xF + intercept_rect
    else:
        slope_q = q / (q - 1)
        intercept_q = -xF / (q - 1)
        # Solve for intersection of rectifying and q-lines
        A = np.array([[slope_rect, -1], [slope_q, -1]])
        b = np.array([-intercept_rect, -intercept_q])
        try:
            solution = np.linalg.solve(A, b)
            x_int, y_int = solution[0], solution[1]
        except np.linalg.LinAlgError:
            x_int, y_int = xF, alpha * xF / (1 + (alpha - 1) * xF)

    if abs(x_int - xB) < 1e-6:
        slope_strip = float('inf')
    else:
        slope_strip = (y_int - xB) / (x_int - xB)
    intercept_strip = y_int - slope_strip * x_int

    # --- 2. Plot Base Diagram ---
    x_eq = np.linspace(0, 1, 400)
    y_eq = alpha * x_eq / (1 + (alpha - 1) * x_eq)
    ax.plot(x_eq, y_eq, 'b-', linewidth=2, label='Equilibrium Curve')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='45° Line')

    x_rect_plot = np.linspace(x_int, xD, 100)
    ax.plot(x_rect_plot, slope_rect * x_rect_plot + intercept_rect, 'r-', linewidth=2,
            label=f'Rectifying Line (R={R:.3f})')
    x_strip_plot = np.linspace(xB, x_int, 100)
    ax.plot(x_strip_plot, slope_strip * x_strip_plot + intercept_strip, 'g-', linewidth=2, label='Stripping Line')

    if abs(q - 1.0) < 1e-6:
        ax.axvline(x=xF, ymin=min(xF, xF), ymax=max(xF, x_int), color='m', linestyle='--', alpha=0.7,
                   label='q-Line (q=1)')
    else:
        x_q_plot = np.linspace(xF, x_int, 100)
        ax.plot(x_q_plot, slope_q * (x_q_plot - xF) + (alpha * xF / (1 + (alpha - 1) * xF)), 'm--', alpha=0.7,
                label=f'q-Line (q={q:.2f})')

    # --- 3. Step Off Stages ---
    x_stages, y_stages = [xD], [xD]
    x_current = xD
    stage_count = 0
    feed_stage = None
    max_stages = 100

    while x_current > xB and stage_count < max_stages:
        stage_count += 1
        y_next_op = (slope_rect * x_current + intercept_rect) if x_current >= x_int else (
                    slope_strip * x_current + intercept_strip)
        x_stages.extend([x_current, x_current])
        y_stages.extend([y_stages[-1], y_next_op])

        denominator = alpha - y_next_op * (alpha - 1)
        if denominator <= 1e-6: break
        x_current = y_next_op / denominator
        x_stages.append(x_current)
        y_stages.append(y_next_op)

        if feed_stage is None and x_current < x_int:
            feed_stage = stage_count

    if feed_stage is None: feed_stage = stage_count
    actual_stages = stage_count
    ax.plot(x_stages, y_stages, 'o-', color='black', markersize=4, linewidth=1.5,
            label=f'Theoretical Stages ({actual_stages})')

    # --- 4. Final Formatting ---
    ax.axvline(x=xD, color='purple', linestyle=':', alpha=0.7, label=f'Distillate (xD={xD:.3f})')
    ax.axvline(x=xB, color='darkorange', linestyle=':', alpha=0.7, label=f'Bottoms (xB={xB:.3f})')
    ax.axvline(x=xF, color='teal', linestyle=':', alpha=0.7, label=f'Feed (xF={xF:.3f})')

    if feed_stage is not None and (2 * feed_stage) < len(x_stages):
        ax.plot(x_stages[2 * feed_stage], y_stages[2 * feed_stage - 1], 's', color='cyan', markersize=10,
                label=f'Feed Stage ({feed_stage})', markeredgecolor='black', markeredgewidth=1.5)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Liquid Mole Fraction (x)', fontsize=12)
    ax.set_ylabel('Vapor Mole Fraction (y)', fontsize=12)

    y_eq_xF = alpha * xF / (1 + (alpha - 1) * xF)
    R_min = (xD - y_eq_xF) / (y_eq_xF - xF) if abs(y_eq_xF - xF) > 1e-6 else float('inf')
    ax.set_title(f'McCabe-Thiele Diagram (R_min={R_min:.3f})', fontsize=14, fontweight='bold')

    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.82, 1])

    return fig, actual_stages, feed_stage, R_min

