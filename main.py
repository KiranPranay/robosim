import numpy as np
import matplotlib.pyplot as plt

# =========================
# CONFIGURATION SECTION
# =========================

# Mass of each arm section (A1..A5) in grams  (from your note)
# A1 is the bottom-most link, A5 is the link holding the gripper.
link_masses_g = np.array([177.1, 104.060, 127.560, 36.240, 73.460])

# Variable link lengths in cm (edit these for your design / experiments)
# L1 is between Motor1–Motor2, L2 between Motor2–Motor3, etc.
link_lengths_cm = np.array([10.0, 12.0, 12.0, 8.0, 6.0])

# Payload at end-effector (battery + gripper + tool etc.) in grams
payload_mass_g = 400.0

# Safety factor for servo selection (1.5–2 is common)
SAFETY_FACTOR = 1.8

# Servo database: stall torque in kg·cm at ~6 V
# (add / edit models as you like)
servo_db = {
    "MG90S": 2.2,
    "MG996R": 11.0,
    "DS3218": 20.0,
    "JX PDI-6221MG": 20.0,
    "LD-3015MG": 15.0,
}

# Optional: your current servo choices for the 6 motors
current_servos = [
    "MG996R",  # Motor 1 (base)
    "MG996R",  # Motor 2
    "MG996R",  # Motor 3
    "MG90S",   # Motor 4
    "MG90S",   # Motor 5
    "MG90S",   # Motor 6 (gripper rotation)
]

# =========================
# COMPUTATION FUNCTIONS
# =========================

def compute_joint_torques_kgcm(link_lengths_cm, link_masses_g, payload_mass_g):
    """
    Computes worst-case required torque at each joint (1..N) in kg·cm
    assuming:
      - planar arm
      - arm fully horizontal (worst case)
      - each link's CoM at its middle
      - payload at very end of last link

    In "kg·cm" units used by servo datasheets:
        torque_kgcm = mass_kg * distance_cm
    so we can work directly in those units.
    """
    n_links = len(link_lengths_cm)
    masses_kg = link_masses_g / 1000.0
    payload_kg = payload_mass_g / 1000.0

    torques = np.zeros(n_links)

    for i in range(n_links):
        torque_i = 0.0
        # Contributions from each link j >= i
        for j in range(i, n_links):
            # distance from joint i to CoM of link j
            distance_cm = np.sum(link_lengths_cm[i:j]) + link_lengths_cm[j] / 2.0
            torque_i += masses_kg[j] * distance_cm
        # Payload at end of last link
        payload_distance_cm = np.sum(link_lengths_cm[i:])
        torque_i += payload_kg * payload_distance_cm
        torques[i] = torque_i

    return torques


def pick_servo_for_torque(required_kgcm, servo_db, safety_factor=1.5):
    """
    Choose the smallest servo that can deliver required_kgcm * safety_factor.
    Returns (servo_name, servo_torque, needed_with_sf).
    """
    needed = required_kgcm * safety_factor
    candidates = sorted(servo_db.items(), key=lambda kv: kv[1])
    for name, torque in candidates:
        if torque >= needed:
            return name, torque, needed
    # If nothing is strong enough, return the strongest and mark under-sized
    name, torque = candidates[-1]
    return name + " (UNDER-SIZED!)", torque, needed

# =========================
# MAIN
# =========================

if __name__ == "__main__":
    required_torques = compute_joint_torques_kgcm(
        link_lengths_cm, link_masses_g, payload_mass_g
    )

    print("Required joint torques (worst case, arm horizontal):")
    for i, t in enumerate(required_torques, start=1):
        print(f"  Joint {i}: {t:.2f} kg·cm")

    # Servo recommendations
    recommended = []
    for i, t in enumerate(required_torques, start=1):
        name, servo_torque, needed = pick_servo_for_torque(
            t, servo_db, SAFETY_FACTOR
        )
        recommended.append((name, servo_torque, needed))
        print(
            f"Joint {i}: need ≥ {needed:.2f} kg·cm with SF={SAFETY_FACTOR}, "
            f"recommended servo: {name} ({servo_torque} kg·cm)"
        )

    # Compare with current servo assignment (MG996R / MG90S from your sketch)
    if len(current_servos) >= len(required_torques):
        print("\n=== Check of CURRENT servo assignment ===")
        for i, (req, servo_name) in enumerate(
            zip(required_torques, current_servos), start=1
        ):
            available = servo_db.get(servo_name, 0.0)
            utilization = req / available * 100 if available > 0 else float("inf")
            print(
                f"Joint {i}: current={servo_name} ({available} kg·cm), "
                f"required={req:.2f} kg·cm, utilization={utilization:.1f}%"
            )

    # =========================
    # VISUALIZATION
    # =========================

    joints = np.arange(1, len(required_torques) + 1)

    required_sf = required_torques * SAFETY_FACTOR
    available = np.array(
        [servo_db[r[0].replace(" (UNDER-SIZED!)", "")] for r in recommended]
    )

    bar_width = 0.35
    plt.figure(figsize=(9, 5))
    plt.bar(
        joints - bar_width / 2,
        required_sf,
        width=bar_width,
        label="Required (with safety factor)",
    )
    plt.bar(
        joints + bar_width / 2,
        available,
        width=bar_width,
        label="Recommended servo rating",
    )

    plt.xlabel("Joint")
    plt.ylabel("Torque (kg·cm)")
    plt.title("Required vs Recommended Servo Torque per Joint")
    plt.xticks(joints)
    plt.legend()
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)

    # Annotate servo names on top
    for i, (x, y) in enumerate(zip(joints + bar_width / 2, available)):
        name = recommended[i][0]
        plt.text(
            x,
            y + 0.3,
            name,
            ha="center",
            va="bottom",
            rotation=45,
            fontsize=8,
        )

    plt.tight_layout()
    plt.show()
