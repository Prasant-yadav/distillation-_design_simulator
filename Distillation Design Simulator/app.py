import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from distillation_core import fenske_equation, underwood_equation
from plotting import create_mccabe_thiele_plot

# Page configuration
st.set_page_config(
    page_title="Distillation Column Simulator",
    page_icon="⚗️",
    layout="wide"
)


# --- UI Functions for Calculators ---

def display_feed_flow_calculator():
    """Renders the UI for the Feed Flow (F) calculator."""
    st.markdown("###### Using Overall Mass Balance (F = D + W)")
    d_in = st.number_input("Distillate Flow (D)", key="d_in_f_overall", min_value=0.0, value=50.0)
    w_in = st.number_input("Bottoms Flow (W)", key="w_in_f_overall", min_value=0.0, value=50.0)
    if st.button("Calculate F", key="calc_f_overall"):
        f_calc = d_in + w_in
        st.success(f"Calculated Feed Flow (F) = **{f_calc:.2f}**")


def display_distillate_flow_calculator():
    """Renders the UI for the Distillate Flow (D) calculator."""
    st.markdown("###### Using Overall Mass Balance (D = F - W)")
    f_in = st.number_input("Feed Flow (F)", key="f_in_d_overall", min_value=0.0, value=100.0)
    w_in = st.number_input("Bottoms Flow (W)", key="w_in_d_overall", min_value=0.0, value=50.0)
    if st.button("Calculate D", key="calc_d_overall"):
        d_calc = f_in - w_in
        st.success(f"Calculated Distillate Flow (D) = **{d_calc:.2f}**")


def display_bottoms_flow_calculator():
    """Renders the UI for the Bottoms Flow (W) calculator."""
    st.markdown("###### Using Overall Mass Balance (W = F - D)")
    f_in = st.number_input("Feed Flow (F)", key="f_in_w_overall", min_value=0.0, value=100.0)
    d_in = st.number_input("Distillate Flow (D)", key="d_in_w_overall", min_value=0.0, value=50.0)
    if st.button("Calculate W", key="calc_w_overall"):
        w_calc = f_in - d_in
        st.success(f"Calculated Bottoms Flow (W) = **{w_calc:.2f}**")


def display_component_flow_calculator():
    """Renders the UI for the D & W from component balance calculator."""
    st.markdown("###### Using Component Mass Balance")
    known_flow = st.selectbox(
        "Which flow rate is known?",
        ('Feed Flow (F)', 'Distillate Flow (D)', 'Bottoms Flow (W)'),
        key="known_flow_selector"
    )

    # Inputs for compositions are always needed
    xf_in = st.number_input("Feed Comp (xF)", key="xf_in_dw_comp", min_value=0.001, max_value=0.999, value=0.5,
                            format="%.3f")
    xd_in = st.number_input("Distillate Comp (xD)", key="xd_in_dw_comp", min_value=0.001, max_value=0.999, value=0.95,
                            format="%.3f")
    xb_in = st.number_input("Bottoms Comp (xB)", key="xb_in_dw_comp", min_value=0.001, max_value=0.999, value=0.05,
                            format="%.3f")

    # Input for the known flow rate
    flow_val = 0.0
    if known_flow == 'Feed Flow (F)':
        flow_val = st.number_input("Known Feed Flow (F)", key="f_in_dw_comp", min_value=0.0, value=100.0)
    elif known_flow == 'Distillate Flow (D)':
        flow_val = st.number_input("Known Distillate Flow (D)", key="d_in_dw_comp", min_value=0.0, value=50.0)
    else:  # Bottoms Flow (W)
        flow_val = st.number_input("Known Bottoms Flow (W)", key="w_in_dw_comp", min_value=0.0, value=50.0)

    if st.button("Calculate Unknown Flows", key="calc_dw_comp"):
        f_calc, d_calc, w_calc = None, None, None
        error_found = False

        if known_flow == 'Feed Flow (F)':
            f_calc = flow_val
            denominator = xd_in - xb_in
            if abs(denominator) > 1e-6:
                d_calc = f_calc * (xf_in - xb_in) / denominator
                w_calc = f_calc - d_calc
            else:
                st.error("Distillate and Bottoms compositions cannot be equal.")
                error_found = True
        elif known_flow == 'Distillate Flow (D)':
            d_calc = flow_val
            denominator = xf_in - xb_in
            if abs(denominator) > 1e-6:
                w_calc = d_calc * (xd_in - xf_in) / denominator
                f_calc = d_calc + w_calc
            else:
                st.error("Feed and Bottoms compositions cannot be equal.")
                error_found = True
        else:  # Bottoms Flow (W)
            w_calc = flow_val
            denominator = xf_in - xd_in
            if abs(denominator) > 1e-6:
                d_calc = w_calc * (xb_in - xf_in) / denominator
                f_calc = d_calc + w_calc
            else:
                st.error("Feed and Distillate compositions cannot be equal.")
                error_found = True

        if not error_found and all(v is not None for v in [f_calc, d_calc, w_calc]):
            st.success(f"Calculated Feed Flow (F) = **{f_calc:.2f}**")
            st.success(f"Calculated Distillate Flow (D) = **{d_calc:.2f}**")
            st.success(f"Calculated Bottoms Flow (W) = **{w_calc:.2f}**")


def display_reflux_ratio_calculator():
    """Renders the UI for the Reflux Ratio (R) calculator."""
    l_in_r = st.number_input("Reflux Molar Flow (L)", key="l_in_r", min_value=0.0, value=100.0, format="%.2f")
    d_in_r = st.number_input("Distillate Molar Flow (D)", key="d_in_r", min_value=0.01, value=50.0, format="%.2f")
    if st.button("Calculate R", key="calc_r"):
        r_calc = l_in_r / d_in_r
        st.success(f"Calculated Reflux Ratio (R) = **{r_calc:.3f}**")


def display_composition_calculator(comp_to_calc):
    """Renders the UI for the composition calculators (xF, xD, xB)."""
    st.markdown("###### Using Component Mass Balance")
    known_flows = st.selectbox(
        "Known Flow Rates:",
        ('Feed (F) and Distillate (D)', 'Feed (F) and Bottoms (W)', 'Distillate (D) and Bottoms (W)'),
        key=f"known_flows_{comp_to_calc}"
    )

    f_flow, d_flow, w_flow = 0.0, 0.0, 0.0
    if known_flows == 'Feed (F) and Distillate (D)':
        f_flow = st.number_input("Feed Flow (F)", key=f"f_flow_fd_{comp_to_calc}", min_value=0.01, value=100.0)
        d_flow = st.number_input("Distillate Flow (D)", key=f"d_flow_fd_{comp_to_calc}", min_value=0.0, value=50.0)
        w_flow = f_flow - d_flow
        if w_flow < 0: st.error("D cannot be greater than F.")
    elif known_flows == 'Feed (F) and Bottoms (W)':
        f_flow = st.number_input("Feed Flow (F)", key=f"f_flow_fw_{comp_to_calc}", min_value=0.01, value=100.0)
        w_flow = st.number_input("Bottoms Flow (W)", key=f"w_flow_fw_{comp_to_calc}", min_value=0.0, value=50.0)
        d_flow = f_flow - w_flow
        if d_flow < 0: st.error("W cannot be greater than F.")
    else:  # 'Distillate (D) and Bottoms (W)'
        d_flow = st.number_input("Distillate Flow (D)", key=f"d_flow_dw_{comp_to_calc}", min_value=0.0, value=50.0)
        w_flow = st.number_input("Bottoms Flow (W)", key=f"w_flow_dw_{comp_to_calc}", min_value=0.0, value=50.0)
        f_flow = d_flow + w_flow

    if comp_to_calc == 'xF':
        xd_in = st.number_input("Distillate Comp (xD)", key="xd_in_xf", min_value=0.0, max_value=1.0, value=0.95)
        xb_in = st.number_input("Bottoms Comp (xB)", key="xb_in_xf", min_value=0.0, max_value=1.0, value=0.05)
        if st.button("Calculate xF", key="calc_xf"):
            if f_flow > 0:
                xf_calc = (d_flow * xd_in + w_flow * xb_in) / f_flow
                st.success(f"Calculated Feed Comp. (xF) = **{xf_calc:.4f}**")
            else:
                st.error("Calculated Feed Flow (F) is zero.")
    elif comp_to_calc == 'xD':
        xf_in = st.number_input("Feed Comp (xF)", key="xf_in_xd", min_value=0.0, max_value=1.0, value=0.5)
        xb_in = st.number_input("Bottoms Comp (xB)", key="xb_in_xd", min_value=0.0, max_value=1.0, value=0.05)
        if st.button("Calculate xD", key="calc_xd"):
            if d_flow > 0:
                xd_calc = (f_flow * xf_in - w_flow * xb_in) / d_flow
                st.success(f"Calculated Distillate Comp. (xD) = **{xd_calc:.4f}**")
            else:
                st.error("Calculated Distillate Flow (D) is zero.")
    elif comp_to_calc == 'xB':
        xf_in = st.number_input("Feed Comp (xF)", key="xf_in_xb", min_value=0.0, max_value=1.0, value=0.5)
        xd_in = st.number_input("Distillate Comp (xD)", key="xd_in_xb", min_value=0.0, max_value=1.0, value=0.95)
        if st.button("Calculate xB", key="calc_xb"):
            if w_flow > 0:
                xb_calc = (f_flow * xf_in - d_flow * xd_in) / w_flow
                st.success(f"Calculated Bottoms Comp. (xB) = **{xb_calc:.4f}**")
            else:
                st.error("Calculated Bottoms Flow (W) is zero.")


def display_mccabe_thiele_rmin_calculator():
    """Renders the UI for the McCabe-Thiele Rmin calculator."""
    st.markdown("###### McCabe-Thiele Intersection Method for Rmin")
    st.latex(r"\frac{R_{min}}{R_{min}+1} = \frac{x_D - y'}{x_D - x'}")
    xd_in_mccabe = st.number_input("Distillate composition (xD)", key="xd_in_rmin_mccabe", min_value=0.001,
                                   max_value=0.999, value=0.95, format="%.3f")
    x_prime = st.number_input("x' (intersection with q-line)", key="x_prime_mccabe", min_value=0.001, max_value=0.999,
                              value=0.45, format="%.3f")
    y_prime = st.number_input("y' (equilibrium at x')", key="y_prime_mccabe", min_value=0.001, max_value=0.999,
                              value=0.70, format="%.3f")

    if st.button("Calculate Rmin (McCabe-Thiele)"):
        numerator = xd_in_mccabe - y_prime
        denominator = xd_in_mccabe - x_prime
        if abs(denominator) > 1e-6:
            slope = numerator / denominator
            if abs(1 - slope) > 1e-6:
                rmin_calc = slope / (1 - slope)
                st.success(f"Calculated Minimum Reflux Ratio (Rmin) = **{rmin_calc:.3f}**")
            else:
                st.error("Calculation failed: (1-slope) is zero, implies infinite reflux.")
        else:
            st.error("Invalid input: xD cannot be equal to x'.")


# --- Main App ---
st.title("⚗️ Distillation Column Simulator")
st.markdown("""
This tool provides a rapid and interactive way to perform preliminary designs for binary distillation columns. 
It combines the Fenske, Underwood, and Gilliland shortcut equations with a visual McCabe-Thiele plot to bridge theory and practical application.
""")

# Sidebar for inputs
st.sidebar.header("Chemical System")
system = st.sidebar.selectbox("Select Chemical System", ["Custom"])
default_alpha = 2.5
default_comp = 0.5

st.sidebar.header("Feed Conditions")
xF = st.sidebar.number_input('Feed Composition (xF)', min_value=0.001, max_value=0.999, value=float(default_comp),
                             step=0.01, format="%.3f")
alpha = st.sidebar.number_input('Relative Volatility (α)', min_value=1.001, max_value=20.0, value=float(default_alpha),
                                step=0.01, format="%.3f")

st.sidebar.header("Feed Quality (q)")
q_method = st.sidebar.selectbox("Specify Feed Quality by:", ("Direct q-value Entry", "Feed Thermal Condition"))
q = 1.0  # Default value
if q_method == "Direct q-value Entry":
    q = st.sidebar.number_input('Feed Quality (q)', min_value=-5.0, max_value=5.0, value=1.0, step=0.01, format="%.2f",
                                help="q=1 (sat. liquid), q=0 (sat. vapor), 0<q<1 (mixed), q>1 (subcooled), q<0 (superheated)")
else:  # Feed Thermal Condition
    condition_type = st.sidebar.selectbox("Select Feed Condition:",
                                          ("Saturated Liquid", "Saturated Vapor", "Mixed Phase", "Subcooled Liquid",
                                           "Superheated Vapor"))
    if condition_type == "Saturated Liquid":
        q = 1.0
    elif condition_type == "Saturated Vapor":
        q = 0.0
    elif condition_type == "Mixed Phase":
        q = st.sidebar.number_input("Liquid Mole Fraction in Feed", min_value=0.00, max_value=1.00, value=0.5,
                                    step=0.01)
    elif condition_type == "Subcooled Liquid":
        q = st.sidebar.number_input("Enter q-value for Subcooled Liquid", min_value=1.001, max_value=5.0, value=1.1,
                                    step=0.1)
    elif condition_type == "Superheated Vapor":
        q = st.sidebar.number_input("Enter q-value for Superheated Vapor", min_value=-5.0, max_value=-0.001, value=-0.1,
                                    step=0.1)

    if condition_type in ["Saturated Liquid", "Saturated Vapor"]:
        st.sidebar.markdown(f"q-value is automatically set to **{q}**.")
    elif condition_type == "Mixed Phase":
        st.sidebar.markdown(f"For a mixed feed, q is the liquid fraction, so q = **{q:.2f}**.")

st.sidebar.header("Product Specifications")
xD = st.sidebar.number_input('Distillate Purity (xD)', min_value=0.001, max_value=0.999, value=0.95, step=0.01,
                             format="%.3f")
xB = st.sidebar.number_input('Bottoms Purity (xB)', min_value=0.001, max_value=0.999, value=0.05, step=0.01,
                             format="%.3f")

st.sidebar.header("Operating Conditions")
R_ratio = st.sidebar.number_input('Reflux Ratio Factor (R/R_min)', min_value=1.05, max_value=5.0, value=1.5, step=0.05,
                                  format="%.2f")

if q == 1.0:
    condition = "Saturated Liquid"
elif q == 0.0:
    condition = "Saturated Vapor"
elif q > 1.0:
    condition = "Subcooled Liquid"
elif q < 0.0:
    condition = "Superheated Vapor"
else:
    condition = "Mixed Phase"
st.sidebar.info(f"**Feed Condition:** {condition} (q={q:.2f})")

validation_errors = []
if xD <= xF: validation_errors.append("❌ Distillate purity (xD) must be greater than feed (xF).")
if xB >= xF: validation_errors.append("❌ Bottoms purity (xB) must be less than feed (xF).")
if xD <= xB: validation_errors.append("❌ Distillate purity (xD) must be greater than bottoms (xB).")
for error in validation_errors: st.sidebar.error(error)

main_container = st.container()

if not validation_errors:
    with main_container:
        with st.spinner('Performing distillation calculations...'):
            N_min = fenske_equation(xD, xB, alpha)
            R_min_calc = underwood_equation(alpha, xF, q, xD)
            R_actual = R_ratio * R_min_calc
            fig, N_actual, feed_stage, R_min_plot = create_mccabe_thiele_plot(xF, xD, xB, R_actual, alpha, q)
            st.success('Simulation Completed!')

            st.subheader("Key Design Results")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Minimum Stages (N_min)", f"{N_min:.2f}")
            col2.metric("Minimum Reflux (R_min)", f"{R_min_plot:.3f}")
            col3.metric("Actual Stages (N_actual)", f"{N_actual}")
            col4.metric("Feed Stage", f"{feed_stage}")
            col5, col6, col7, col8 = st.columns(4)
            col5.metric("Actual Reflux (R)", f"{R_actual:.3f}")
            col6.metric("Reflux Factor", f"{R_ratio:.2f}")
            if N_actual > 0 and feed_stage is not None and feed_stage > 0:
                col7.metric("Rectifying Stages", f"{feed_stage - 1}")
                col8.metric("Stripping Stages", f"{N_actual - feed_stage + 1}")

            plot_col, summary_col = st.columns([2, 1])
            with plot_col:
                st.subheader("📊 McCabe-Thiele Diagram")
                st.pyplot(fig)
            with summary_col:
                st.subheader("💡 Results Summary")
                summary_text = f"""
                The distillation column design for the **{system}** separation requires:
                - **{N_actual} theoretical stages** (including reboiler).
                - An operating reflux ratio of **{R_actual:.3f}**.
                - Feed should be introduced at **stage {feed_stage}**.
                - This design achieves **{xD * 100:.1f}% purity** in the distillate and **{xB * 100:.1f}% purity** in the bottoms.
                - The feed is a **{condition.lower()}**.
                """
                st.info(summary_text)

        st.markdown("---")
        st.subheader("⚙️ Calculation Utilities")
        with st.expander("Show Calculators", expanded=False):
            util_col1, util_col2 = st.columns(2)
            with util_col1:
                st.markdown("#### Mass Balance & Process Calculator")
                to_calculate = st.radio(
                    "Select value to calculate:",
                    ('Feed Flow (F)', 'Distillate Flow (D)', 'Bottoms Flow (W)',
                     'Distillate & Bottoms Flows (D & W)', 'Reflux Ratio (R)',
                     'Feed Comp. (xF)', 'Distillate Comp. (xD)', 'Bottoms Comp. (xB)'),
                    horizontal=True, key="calc_selector"
                )

                if to_calculate == 'Feed Flow (F)':
                    display_feed_flow_calculator()
                elif to_calculate == 'Distillate Flow (D)':
                    display_distillate_flow_calculator()
                elif to_calculate == 'Bottoms Flow (W)':
                    display_bottoms_flow_calculator()
                elif to_calculate == 'Distillate & Bottoms Flows (D & W)':
                    display_component_flow_calculator()
                elif to_calculate == 'Reflux Ratio (R)':
                    display_reflux_ratio_calculator()
                elif to_calculate == 'Feed Comp. (xF)':
                    display_composition_calculator('xF')
                elif to_calculate == 'Distillate Comp. (xD)':
                    display_composition_calculator('xD')
                elif to_calculate == 'Bottoms Comp. (xB)':
                    display_composition_calculator('xB')

                st.markdown("---")
                display_mccabe_thiele_rmin_calculator()

            with util_col2:
                st.markdown("#### Operating Line Properties")
                st.write("Properties based on the main simulation inputs above.")

                st.markdown("**Rectifying Line:** `y = m*x + c`")
                slope_rect = R_actual / (R_actual + 1)
                intercept_rect = xD / (R_actual + 1)
                st.latex(
                    f"m_R = \\frac{{R}}{{R+1}} = \\frac{{{R_actual:.3f}}}{{{R_actual + 1:.3f}}} = {slope_rect:.4f}")
                st.latex(
                    f"c_R = \\frac{{x_D}}{{R+1}} = \\frac{{{xD:.3f}}}{{{R_actual + 1:.3f}}} = {intercept_rect:.4f}")

                st.markdown("**q-Line:** `y = m*x + c`")
                if q == 1.0:
                    st.info("The q-line is a vertical line at x = xF (slope is infinite).")
                    st.latex("m_q = \\infty")
                else:
                    slope_q = q / (q - 1)
                    intercept_q = -xF / (q - 1)
                    st.latex(f"m_q = \\frac{{q}}{{q-1}} = \\frac{{{q:.2f}}}{{{q - 1:.2f}}} = {slope_q:.4f}")
                    st.latex(f"c_q = \\frac{{-x_F}}{{q-1}} = \\frac{{{-xF:.3f}}}{{{q - 1:.2f}}} = {intercept_q:.4f}")

                st.markdown("**Stripping Line:** `y = m*x + c`")
                if q == 1.0:
                    x_int = xF
                    y_int = slope_rect * xF + intercept_rect
                else:
                    slope_q = q / (q - 1) if q != 1 else float('inf')
                    intercept_q = -xF / (q - 1) if q != 1 else 0
                    if abs(slope_rect - slope_q) > 1e-6:
                        x_int = (intercept_q - intercept_rect) / (slope_rect - slope_q)
                        y_int = slope_rect * x_int + intercept_rect
                    else:
                        x_int, y_int = xF, xF

                if abs(x_int - xB) > 1e-6:
                    slope_strip = (y_int - xB) / (x_int - xB)
                    intercept_strip = y_int - slope_strip * x_int
                    st.latex(
                        f"m_S = \\frac{{y_{{int}} - x_B}}{{x_{{int}} - x_B}} = \\frac{{{y_int:.3f} - {xB:.3f}}}{{{x_int:.3f} - {xB:.3f}}} = {slope_strip:.4f}")
                    st.latex(
                        f"c_S = y_{{int}} - m_S \\cdot x_{{int}} = {y_int:.3f} - ({slope_strip:.3f})({x_int:.3f}) = {intercept_strip:.4f}")
                else:
                    st.info("Stripping line slope cannot be determined (intersection is at xB).")

else:
    with main_container:
        st.warning("Please fix the input errors in the sidebar to run the simulation.")

if 'N_actual' not in locals():
    st.info("""
     ### 💡 Welcome to the Distillation Simulator!
     1. **Select** a chemical system or use "Custom" settings in the sidebar.
     2. **Adjust** the values for feed, product, and operating conditions.
     3. The results and McCabe-Thiele plot will update automatically.
     4. Review the key design results and summary.
     """)

