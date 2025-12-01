import streamlit as st
import numpy as np
import plotly.graph_objects as go
import json
import os
from fpdf import FPDF
import matplotlib.pyplot as plt
import tempfile

# =========================
# CONFIGURATION & DEFAULTS
# =========================

CONFIG_FILE = "motors-config.json"
SPECS_FILE = "specs.json"

def load_json(file_path):
    if not os.path.exists(file_path):
        return {}
    with open(file_path, "r") as f:
        return json.load(f)

def save_json(file_path, data):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

st.set_page_config(page_title="Robotic Arm Simulator", layout="wide")

# Load Data
if 'motor_db' not in st.session_state:
    st.session_state.motor_db = load_json(CONFIG_FILE)
if 'specs_db' not in st.session_state:
    st.session_state.specs_db = load_json(SPECS_FILE)

# Initialize Session State Defaults for Widgets
defaults = {
    "len_x": 100.0,
    "len_y": 120.0,
    "len_z": 120.0,
    "len_gap": 35.0,
    "len_a": 60.0,
    "mass_x": 177.1,
    "mass_y": 104.06,
    "mass_z": 127.56,
    "mass_a": 36.24,
    "pay_mass": 200.0,
    "safe_fact": 1.8,
    "m_select_0": "MG996R",
    "m_select_1": "MG996R",
    "m_select_2": "MG996R",
    "m_select_3": "MG90S",
    "m_select_4": "MG90S"
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

st.title("🤖 Robotic Arm Torque Calculator & Visualizer")

# =========================
# SPECS MANAGEMENT (SAVE/LOAD)
# =========================

with st.expander("💾 Manage Configurations (Save / Load Specs)"):
    col_s1, col_s2, col_s3 = st.columns([1, 1, 1])
    
    # SAVE
    with col_s1:
        st.markdown("#### Save Current Specs")
        save_name = st.text_input("Configuration Name", key="spec_save_name")
        if st.button("Save Configuration"):
            if save_name:
                # Gather current state
                current_spec = {
                    "lx": st.session_state["len_x"],
                    "ly": st.session_state["len_y"],
                    "lz": st.session_state["len_z"],
                    "l_gap": st.session_state["len_gap"],
                    "la": st.session_state["len_a"],
                    "mx": st.session_state["mass_x"],
                    "my": st.session_state["mass_y"],
                    "mz": st.session_state["mass_z"],
                    "ma": st.session_state["mass_a"],
                    "payload": st.session_state["pay_mass"],
                    "sf": st.session_state["safe_fact"],
                    "motors": [st.session_state.get(f"m_select_{i}") for i in range(5)]
                }
                st.session_state.specs_db[save_name] = current_spec
                save_json(SPECS_FILE, st.session_state.specs_db)
                st.success(f"Saved '{save_name}'")
            else:
                st.error("Enter a name")

    # LOAD
    with col_s2:
        st.markdown("#### Load Configuration")
        load_name = st.selectbox("Select Config", [""] + list(st.session_state.specs_db.keys()), key="spec_load_sel")
        if st.button("Load Configuration"):
            if load_name and load_name in st.session_state.specs_db:
                spec = st.session_state.specs_db[load_name]
                # Update session state
                st.session_state["len_x"] = float(spec.get("lx", 100.0))
                st.session_state["len_y"] = float(spec.get("ly", 120.0))
                st.session_state["len_z"] = float(spec.get("lz", 120.0))
                st.session_state["len_gap"] = float(spec.get("l_gap", 35.0))
                st.session_state["len_a"] = float(spec.get("la", 60.0))
                st.session_state["mass_x"] = float(spec.get("mx", 177.1))
                st.session_state["mass_y"] = float(spec.get("my", 104.06))
                st.session_state["mass_z"] = float(spec.get("mz", 127.56))
                st.session_state["mass_a"] = float(spec.get("ma", 36.24))
                st.session_state["pay_mass"] = float(spec.get("payload", 200.0))
                st.session_state["safe_fact"] = float(spec.get("sf", 1.8))
                
                # Load motors
                saved_motors = spec.get("motors", [])
                for i, m_name in enumerate(saved_motors):
                    if i < 5:
                        st.session_state[f"m_select_{i}"] = m_name
                
                st.success(f"Loaded '{load_name}'")
                st.rerun()

    # DELETE
    with col_s3:
        st.markdown("#### Delete Configuration")
        del_spec_name = st.selectbox("Select to Delete", [""] + list(st.session_state.specs_db.keys()), key="spec_del_sel")
        if del_spec_name:
            if st.button("Delete Config", type="primary"):
                del st.session_state.specs_db[del_spec_name]
                save_json(SPECS_FILE, st.session_state.specs_db)
                st.success(f"Deleted '{del_spec_name}'")
                st.rerun()

# =========================
# MOTOR DATABASE MANAGER
# =========================

with st.expander("🛠️ Manage Motor Database (Add / Edit / Delete)"):
    col_m1, col_m2, col_m3 = st.columns([1, 1, 1])
    
    # ADD NEW
    with col_m1:
        st.markdown("#### Add New Motor")
        new_name = st.text_input("Model Name")
        new_torque = st.number_input("Stall Torque (6.6v) (kg.cm)", min_value=0.0, step=0.1, key="new_t")
        new_weight = st.number_input("Weight (g)", min_value=0.0, step=1.0, key="new_w")
        if st.button("Add Motor"):
            if new_name and new_name not in st.session_state.motor_db:
                st.session_state.motor_db[new_name] = {"torque": new_torque, "weight": new_weight}
                save_json(CONFIG_FILE, st.session_state.motor_db)
                st.success(f"Added {new_name}")
                st.rerun()
            elif new_name in st.session_state.motor_db:
                st.error("Motor already exists!")
            else:
                st.error("Name required")

    # EDIT EXISTING
    with col_m2:
        st.markdown("#### Edit Motor")
        edit_name = st.selectbox("Select to Edit", [""] + list(st.session_state.motor_db.keys()), key="edit_sel")
        if edit_name:
            curr_vals = st.session_state.motor_db[edit_name]
            edit_torque = st.number_input("Stall Torque (6.6v) (kg.cm)", value=curr_vals['torque'], step=0.1, key="edit_t")
            edit_weight = st.number_input("Weight (g)", value=curr_vals['weight'], step=1.0, key="edit_w")
            if st.button("Update Motor"):
                st.session_state.motor_db[edit_name] = {"torque": edit_torque, "weight": edit_weight}
                save_json(CONFIG_FILE, st.session_state.motor_db)
                st.success(f"Updated {edit_name}")
                st.rerun()

    # DELETE
    with col_m3:
        st.markdown("#### Delete Motor")
        del_name = st.selectbox("Select to Delete", [""] + list(st.session_state.motor_db.keys()), key="del_sel")
        if del_name:
            if st.button("Delete Motor", type="primary"):
                del st.session_state.motor_db[del_name]
                save_json(CONFIG_FILE, st.session_state.motor_db)
                st.success(f"Deleted {del_name}")
                st.rerun()

# =========================
# SIDEBAR CONTROLS
# =========================

st.sidebar.header("⚙️ Arm Configuration")

# 1. Link Lengths (mm)
st.sidebar.subheader("Link Dimensions (mm)")
st.sidebar.caption("Based on your sketch labels (x, y, z, a)")
lx = st.sidebar.number_input("x (Base Height)", step=5.0, key="len_x")
ly = st.sidebar.number_input("y (Shoulder -> Elbow)", step=5.0, key="len_y")
lz = st.sidebar.number_input("z (Elbow -> Wrist M4)", step=5.0, key="len_z")
l_gap = st.sidebar.number_input("Wrist Span (M4 -> M5)", step=1.0, help="Distance between Motor 4 and Motor 5", key="len_gap")
la = st.sidebar.number_input("a (M5 -> Suction Tip)", step=5.0, key="len_a")

# 2. Link Masses (g)
st.sidebar.subheader("Link Masses (g)")
mx = st.sidebar.number_input("Link x Mass", step=10.0, key="mass_x")
my = st.sidebar.number_input("Link y Mass", step=10.0, key="mass_y")
mz = st.sidebar.number_input("Link z Mass", step=10.0, key="mass_z")
ma = st.sidebar.number_input("Link a Mass", step=5.0, key="mass_a")

# 3. Motor Configuration (5 Motors)
st.sidebar.subheader("Motor Configuration")
st.sidebar.caption("Select from Database")

motor_db = st.session_state.motor_db
motor_names = list(motor_db.keys())
motor_selections = []
motor_masses = []

motor_roles = [
    "M1: Base Rotation (Yaw)",
    "M2: Shoulder (Pitch)",
    "M3: Elbow (Pitch)",
    "M4: Wrist 1 (Roll/Pitch)",
    "M5: Wrist 2 (Pitch)"
]

for i in range(5):
    col1, col2 = st.sidebar.columns([2, 1])
    with col1:
        # Use key for session state persistence, removed index
        m_name = st.selectbox(motor_roles[i], motor_names, key=f"m_select_{i}")
    with col2:
        # Display weight from DB
        db_weight = motor_db.get(m_name, {}).get('weight', 0.0)
        st.text(f"Weight:\n{db_weight} g")
    
    motor_selections.append(m_name)
    motor_masses.append(db_weight)

motor_masses = np.array(motor_masses)
payload_mass = st.sidebar.number_input("Payload Mass (g)", step=10.0, key="pay_mass")
safety_factor = st.sidebar.slider("Safety Factor", 1.0, 3.0, step=0.1, key="safe_fact")

# =========================
# CALCULATION LOGIC
# =========================

def compute_torques_custom(lx, ly, lz, l_gap, la, mx, my, mz, ma, motor_masses, payload_g):
    """
    Computes torque with M4 bearing load.
    """
    Ly = ly / 10.0
    Lz = lz / 10.0
    Lgap = l_gap / 10.0
    La = la / 10.0
    
    My = my / 1000.0
    Mz = mz / 1000.0
    Ma = ma / 1000.0
    
    Mm3 = motor_masses[2] / 1000.0
    Mm4 = motor_masses[3] / 1000.0
    Mm5 = motor_masses[4] / 1000.0
    
    Mp = payload_g / 1000.0
    
    torques = np.zeros(5)
    
    # --- M5 (Wrist 2) ---
    t5 = (Ma * (La / 2.0)) + (Mp * La)
    torques[4] = t5
    
    # --- M4 (Wrist 1) ---
    t4 = (Mm5 * Lgap) + \
         (Ma * (Lgap + La / 2.0)) + \
         (Mp * (Lgap + La))
    torques[3] = t4
    
    # --- M3 (Elbow) ---
    t3 = (Mz * (Lz / 2.0)) + \
         (Mm4 * Lz) + \
         (Mm5 * (Lz + Lgap)) + \
         (Ma * (Lz + Lgap + La / 2.0)) + \
         (Mp * (Lz + Lgap + La))
    torques[2] = t3
    
    # --- M2 (Shoulder) ---
    t2 = (My * (Ly / 2.0)) + \
         (Mm3 * Ly) + \
         (Mz * (Ly + Lz / 2.0)) + \
         (Mm4 * (Ly + Lz)) + \
         (Mm5 * (Ly + Lz + Lgap)) + \
         (Ma * (Ly + Lz + Lgap + La / 2.0)) + \
         (Mp * (Ly + Lz + Lgap + La))
    torques[1] = t2
    
    # --- M1 (Base) ---
    torques[0] = 0.0
    
    return torques

torques = compute_torques_custom(lx, ly, lz, l_gap, la, mx, my, mz, ma, motor_masses, payload_mass)

# =========================
# VISUALIZATION (2D)
# =========================

col_viz, col_data = st.columns([2, 1])

with col_viz:
    st.subheader("2D Visualization (Side View)")
    
    x_coords = [0, 0, ly, ly+lz, ly+lz+l_gap, ly+lz+l_gap+la]
    y_coords = [0, lx, lx, lx, lx, lx]
    
    fig = go.Figure()
    
    # Links
    fig.add_trace(go.Scatter(
        x=x_coords, y=y_coords,
        mode='lines+markers+text',
        line=dict(color='#00CC96', width=6),
        marker=dict(size=12, color='#EF553B', symbol='circle'),
        text=["Base", "M2", "M3", "M4", "M5", "Tip"],
        textposition="top center",
        name='Arm Structure'
    ))
    
    # Annotations
    fig.add_annotation(x=0, y=lx/2, text=f"x", xshift=-20, showarrow=False)
    fig.add_annotation(x=ly/2, y=lx, text=f"y", yshift=20, showarrow=False)
    fig.add_annotation(x=ly + lz/2, y=lx, text=f"z", yshift=20, showarrow=False)
    fig.add_annotation(x=ly + lz + l_gap/2, y=lx, text=f"gap", yshift=20, showarrow=False)
    fig.add_annotation(x=ly + lz + l_gap + la/2, y=lx, text=f"a", yshift=20, showarrow=False)

    fig.update_layout(
        title="Worst-Case Extension (Horizontal)",
        xaxis=dict(title="Distance (mm)", zeroline=True),
        yaxis=dict(title="Height (mm)", scaleanchor="x", scaleratio=1),
        height=400,
        plot_bgcolor='#1E1E1E',
        paper_bgcolor='#1E1E1E',
        font=dict(color='white')
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col_data:
    st.subheader("Torque Analysis")
    
    lifting_indices = [1, 2, 3, 4] # M2, M3, M4, M5
    
    st.write("#### Lifting Motors")
    for idx in lifting_indices:
        m_name = motor_selections[idx]
        req = torques[idx]
        req_sf = req * safety_factor
        avail = motor_db.get(m_name, {}).get('torque', 0.0)
        
        status = "✅ OK" if avail >= req_sf else "❌ OVERLOAD"
        
        st.markdown(f"""
        **Motor {idx+1}** ({m_name})
        - Required: `{req:.2f}` kg.cm
        - w/ SF: `{req_sf:.2f}` kg.cm
        - Available: `{avail}` kg.cm
        - Status: {status}
        """)
        
    st.write("---")
    st.write("#### Base Motor")
    m_name = motor_selections[0]
    avail = motor_db.get(m_name, {}).get('torque', 0.0)
    st.markdown(f"""
    **Motor 1** ({m_name})
    - Static Torque: `~0` (Inertia only)
    - Available: `{avail}` kg.cm
    """)

# =========================
# FORMULAS & EXPORT
# =========================

st.markdown("---")
st.subheader("📚 Physics & Formulas")

with st.expander("Show Detailed Torque Formulas"):
    st.markdown(r"""
    The torque required at each joint is calculated by summing the moments of all components (links, motors, payload) that the joint must lift against gravity.
    We assume the **Worst Case Scenario**: The arm is fully extended horizontally.
    
    **Variables:**
    - $L_y, L_z, L_{gap}, L_a$: Lengths of links (cm)
    - $M_y, M_z, M_a$: Masses of links (kg)
    - $M_{m3}, M_{m4}, M_{m5}$: Masses of motors (kg)
    - $M_p$: Payload mass (kg)
    - $g$: Gravity (implicitly handled by using kg.cm units directly if we consider mass as weight force in kgf-ish, but here we sum moments in kg*cm directly).
    
    **Formula for Motor 5 (Wrist Pitch):**
    $$T_5 = M_a \cdot \frac{L_a}{2} + M_p \cdot L_a$$
    
    **Formula for Motor 4 (Wrist Roll/Pitch Support):**
    $$T_4 = M_{m5} \cdot L_{gap} + M_a \cdot (L_{gap} + \frac{L_a}{2}) + M_p \cdot (L_{gap} + L_a)$$
    
    **Formula for Motor 3 (Elbow):**
    $$T_3 = M_z \cdot \frac{L_z}{2} + M_{m4} \cdot L_z + M_{m5} \cdot (L_z + L_{gap}) + M_a \cdot (L_z + L_{gap} + \frac{L_a}{2}) + M_p \cdot (L_z + L_{gap} + L_a)$$
    
    **Formula for Motor 2 (Shoulder):**
    $$T_2 = M_y \frac{L_y}{2} + M_{m3} L_y + M_z (L_y + \frac{L_z}{2}) + M_{m4} (L_y + L_z) + M_{m5} (L_y + L_z + L_{gap}) + M_a (L_y + L_z + L_{gap} + \frac{L_a}{2}) + M_p (L_y + L_z + L_{gap} + L_a)$$
    """)

st.subheader("📄 Report Generation")

def create_pdf():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Robotic Arm Simulation Report", ln=True, align='C')
    pdf.ln(10)
    
    # Configuration
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="1. Configuration", ln=True)
    pdf.set_font("Arial", size=10)
    
    config_text = f"""
    Lengths (mm): x={lx}, y={ly}, z={lz}, gap={l_gap}, a={la}
    Link Masses (g): x={mx}, y={my}, z={mz}, a={ma}
    Payload: {payload_mass} g
    Safety Factor: {safety_factor}
    """
    pdf.multi_cell(0, 5, config_text)
    pdf.ln(5)
    
    # Motor Results
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="2. Motor Analysis", ln=True)
    pdf.set_font("Arial", size=10)
    
    # Header
    pdf.cell(30, 7, "Motor", 1)
    pdf.cell(40, 7, "Model", 1)
    pdf.cell(35, 7, "Required (kg.cm)", 1)
    pdf.cell(35, 7, "Available (kg.cm)", 1)
    pdf.cell(30, 7, "Status", 1)
    pdf.ln()
    
    for i in range(5):
        m_name = motor_selections[i]
        req = torques[i] * safety_factor
        avail = motor_db.get(m_name, {}).get('torque', 0.0)
        status = "OK" if avail >= req else "OVERLOAD"
        
        pdf.cell(30, 7, f"M{i+1}", 1)
        pdf.cell(40, 7, m_name, 1)
        pdf.cell(35, 7, f"{req:.2f}", 1)
        pdf.cell(35, 7, f"{avail:.1f}", 1)
        pdf.cell(30, 7, status, 1)
        pdf.ln()
        
    pdf.ln(10)
    
    # 2D Diagram (Generate Matplotlib image)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="3. 2D Diagram", ln=True)
    
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
        # Create matplotlib figure matching the plotly one roughly
        plt.figure(figsize=(8, 4))
        x_pts = [0, 0, ly, ly+lz, ly+lz+l_gap, ly+lz+l_gap+la]
        y_pts = [0, lx, lx, lx, lx, lx]
        plt.plot(x_pts, y_pts, 'o-', linewidth=3, markersize=8, color='#00CC96')
        plt.title("Arm Extension (Side View)")
        plt.xlabel("Distance (mm)")
        plt.ylabel("Height (mm)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(tmpfile.name)
        plt.close()
        
        pdf.image(tmpfile.name, x=10, w=170)
        # Cleanup
        try:
            os.unlink(tmpfile.name)
        except:
            pass

    # Formulas
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="4. Torque Formulas (Worst Case Horizontal)", ln=True)
    pdf.set_font("Arial", size=10)
    
    formulas_text = """
    Variables:
    Ly, Lz, Lgap, La: Lengths (cm)
    My, Mz, Ma: Link Masses (kg)
    Mm3, Mm4, Mm5: Motor Masses (kg)
    Mp: Payload (kg)

    Motor 5 (Wrist Pitch):
    T5 = Ma * (La/2) + Mp * La

    Motor 4 (Wrist Roll/Pitch Support):
    T4 = Mm5 * Lgap + Ma * (Lgap + La/2) + Mp * (Lgap + La)

    Motor 3 (Elbow):
    T3 = Mz * (Lz/2) + Mm4 * Lz + Mm5 * (Lz + Lgap) + 
         Ma * (Lz + Lgap + La/2) + Mp * (Lz + Lgap + La)

    Motor 2 (Shoulder):
    T2 = My*(Ly/2) + Mm3*Ly + Mz*(Ly + Lz/2) + Mm4*(Ly + Lz) + 
         Mm5*(Ly + Lz + Lgap) + Ma*(Ly + Lz + Lgap + La/2) + 
         Mp*(Ly + Lz + Lgap + La)
    """
    pdf.multi_cell(0, 5, formulas_text)

    # Conclusion
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="5. Conclusion & Recommendations", ln=True)
    pdf.set_font("Arial", size=10)

    failures = []
    for i in range(5):
        m_name = motor_selections[i]
        req = torques[i] * safety_factor
        avail = motor_db.get(m_name, {}).get('torque', 0.0)
        if avail < req:
            failures.append(f"Motor {i+1} ({m_name})")

    if not failures:
        concl = (f"SUCCESS: The proposed design meets all torque requirements with a Safety Factor of {safety_factor}. "
                 "All selected motors operate within their rated stall torque limits.")
    else:
        concl = (f"FAILURE: The design is UNDERPOWERED. The following motors are insufficient for the load: {', '.join(failures)}. "
                 "Please select higher torque motors, reduce link lengths/masses, or lower the payload.")

    power_note = ("\n\nElectrical Considerations:\n"
                  "- Voltage: Ensure a stable power supply of 6.6V (as per stall torque specs).\n"
                  "- Current: High torque loads draw significant current. If motors are near their limit, "
                  "expect peak currents >2A per motor. Ensure your power supply can handle the total peak current "
                  "to prevent voltage sag, which can cause system instability.")

    pdf.multi_cell(0, 5, concl + power_note)

    return pdf.output(dest='S').encode('latin-1')

if st.button("Generate PDF Report"):
    pdf_bytes = create_pdf()
    st.download_button(
        label="Download PDF",
        data=pdf_bytes,
        file_name="robotic_arm_report.pdf",
        mime="application/pdf"
    )
