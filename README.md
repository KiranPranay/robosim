# RoboSIM 🤖

**Robotic Arm Torque Calculator & Visualizer**

RoboSIM is an interactive Streamlit application designed to assist in the design and analysis of 5-DOF robotic arms. It allows users to configure link dimensions, masses, and motor specifications to calculate the required torque for each joint under worst-case loading conditions (horizontal extension).

![RoboSIM Screenshot](screenshot.png)

## Features 🚀

*   **Interactive Configuration**: Adjust link lengths (mm) and masses (g) via a user-friendly sidebar.
*   **Motor Database**: Manage your servo inventory (Add/Edit/Delete) with specifications for Stall Torque (6.6V) and Weight.
*   **Specs Management**: Save and Load multiple arm configurations (`specs.json`) to quickly iterate on designs.
*   **Real-time Torque Analysis**: Instantly calculates required torque vs. available torque for all 5 motors (Base, Shoulder, Elbow, Wrist 1, Wrist 2).
*   **2D Visualization**: Visualizes the arm structure in its worst-case horizontal extension using Plotly.
*   **PDF Report Generation**: Exports a professional PDF report containing:
    *   Configuration Summary
    *   Motor Analysis (Pass/Fail status)
    *   2D Diagram
    *   Detailed Physics Formulas
    *   Conclusion & Electrical Recommendations

## Physics & Calculations 📐

The application calculates torque based on the **Worst Case Scenario**, where the arm is fully extended horizontally. It accounts for:
*   Mass of each link.
*   Mass of each motor (acting as a load on previous joints).
*   Payload mass at the end effector.
*   Lever arm distances from each joint.

## Installation 🛠️

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/KiranPranay/robosim.git
    cd robosim
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application**:
    ```bash
    streamlit run app.py
    ```

## Usage 💡

1.  **Configure Arm**: Use the sidebar to set dimensions `x, y, z, gap, a` and their corresponding masses.
2.  **Select Motors**: Choose motors from the dropdowns. If a motor isn't listed, add it via the "Manage Motor Database" expander.
3.  **Analyze**: Check the "Torque Analysis" section. Green checks ✅ indicate the motor is sufficient; Red crosses ❌ indicate overload.
4.  **Save/Load**: Use the "Manage Configurations" expander to save your current design.
5.  **Export**: Click "Generate PDF Report" to download a comprehensive summary of your design.

## License 📄

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
*Created by @kiranranay*
