import streamlit as st
import numpy as np
import plotly.graph_objects as go
import json
import os

# =========================
# CONFIGURATION & DEFAULTS
# =========================

CONFIG_FILE = "motors-config.json"

def load_motors():
    if not os.path.exists(CONFIG_FILE):
        # Default fallback if file missing
        return {
            "MG996R": {"torque": 11.0, "weight": 55.0},
            "MG90S": {"torque": 2.2, "weight": 13.4}
        }
    with open(CONFIG_FILE, "r") as f:
        return json.load(f)

def save_motors(data):
    with open(CONFIG_FILE, "w") as f:
        json.dump(data, f, indent=4)

st.set_page_config(page_title="Robotic Arm Simulator", layout="wide")

# Load Data
if 'motor_db' not in st.session_state:
    st.session_state.motor_db = load_motors()

st.title("🤖 Robotic Arm Torque Calculator & Visualizer")

# =========================
# MOTOR DATABASE MANAGER
# =========================

with st.expander("🛠️ Manage Motor Database (Add / Edit / Delete)"):
    col_m1, col_m2, col_m3 = st.columns([1, 1, 1])
    
    # ADD NEW
    with col_m1:
        st.markdown("#### Add New Motor")
        new_name = st.text_input("Model Name")
        new_torque = st.number_input("Torque (kg.cm)", min_value=0.0, step=0.1, key="new_t")
        new_weight = st.number_input("Weight (g)", min_value=0.0, step=1.0, key="new_w")
        if st.button("Add Motor"):
            if new_name and new_name not in st.session_state.motor_db:
                st.session_state.motor_db[new_name] = {"torque": new_torque, "weight": new_weight}
                save_motors(st.session_state.motor_db)
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
            edit_torque = st.number_input("Torque (kg.cm)", value=curr_vals['torque'], step=0.1, key="edit_t")
            edit_weight = st.number_input("Weight (g)", value=curr_vals['weight'], step=1.0, key="edit_w")
            if st.button("Update Motor"):
                st.session_state.motor_db[edit_name] = {"torque": edit_torque, "weight": edit_weight}
                save_motors(st.session_state.motor_db)
                st.success(f"Updated {edit_name}")
                st.rerun()

    # DELETE
    with col_m3:
        st.markdown("#### Delete Motor")
        del_name = st.selectbox("Select to Delete", [""] + list(st.session_state.motor_db.keys()), key="del_sel")
        if del_name:
            if st.button("Delete Motor", type="primary"):
                del st.session_state.motor_db[del_name]
                save_motors(st.session_state.motor_db)
                st.success(f"Deleted {del_name}")
                st.rerun()

# =========================
# SIDEBAR CONTROLS
# =========================

st.sidebar.header("⚙️ Arm Configuration")

# 1. Link Lengths (mm)
st.sidebar.subheader("Link Dimensions (mm)")
st.sidebar.caption("Based on your sketch labels (x, y, z, a)")
lx = st.sidebar.number_input("x (Base Height)", value=100.0, step=5.0)
ly = st.sidebar.number_input("y (Shoulder -> Elbow)", value=120.0, step=5.0)
lz = st.sidebar.number_input("z (Elbow -> Wrist M4)", value=120.0, step=5.0)
l_gap = st.sidebar.number_input("Wrist Span (M4 -> M5)", value=35.0, step=1.0, help="Distance between Motor 4 and Motor 5")
la = st.sidebar.number_input("a (M5 -> Suction Tip)", value=60.0, step=5.0)

# 2. Link Masses (g)
st.sidebar.subheader("Link Masses (g)")
mx = st.sidebar.number_input("Link x Mass", value=177.1, step=10.0)
my = st.sidebar.number_input("Link y Mass", value=104.06, step=10.0)
mz = st.sidebar.number_input("Link z Mass", value=127.56, step=10.0)
ma = st.sidebar.number_input("Link a Mass", value=36.24, step=5.0)

# 3. Motor Configuration (5 Motors)
st.sidebar.subheader("Motor Configuration")
st.sidebar.caption("Select from Database")

motor_db = st.session_state.motor_db
motor_names = list(motor_db.keys())
motor_selections = []
motor_masses = []

# Default selections
# Try to find defaults if they exist, else first item
def get_idx(name):
    try:
        return motor_names.index(name)
    except:
        return 0

default_names = ["MG996R", "MG996R", "MG996R", "MG90S", "MG90S"]
default_indices = [get_idx(n) for n in default_names]

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
        m_name = st.selectbox(motor_roles[i], motor_names, index=default_indices[i] if i < len(default_indices) else 0, key=f"m_select_{i}")
    with col2:
        # Read-only display of weight, or allow override?
        # Let's allow override but default to DB value
        db_weight = motor_db[m_name]['weight']
        m_weight = st.number_input(f"Wt(g)##{i}", value=db_weight, step=1.0, key=f"m_weight_{i}")
    
    motor_selections.append(m_name)
    motor_masses.append(m_weight)

motor_masses = np.array(motor_masses)

# 4. Payload
st.sidebar.subheader("Payload")
payload_mass = st.sidebar.number_input("Payload Mass (g)", value=200.0, step=10.0)
safety_factor = st.sidebar.slider("Safety Factor", 1.0, 3.0, 1.8, 0.1)

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
        
        # Get torque from DB (or use default if missing)
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
