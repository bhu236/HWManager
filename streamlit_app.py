import streamlit as st
import glob
from pathlib import Path

# ----------------------------
# Page configuration
# ----------------------------
st.set_page_config(page_title="HW Manager", layout="wide")
st.title("📚 HW Manager - Chatbot Assignments")

# ----------------------------
# Find HW files
# ----------------------------
CURRENT_DIR = Path(__file__).parent
hw_files = sorted(glob.glob(str(CURRENT_DIR / "HW*.py")))

if not hw_files:
    st.error("No HW files found.")
    st.stop()

# Map HW names to file paths
hw_mapping = {Path(f).stem: f for f in hw_files}

# ----------------------------
# Sidebar selection
# ----------------------------
selected_hw = st.sidebar.selectbox("Select HW", list(hw_mapping.keys()))

file_path = Path(hw_mapping[selected_hw])

# ----------------------------
# Run selected HW file
# ----------------------------
if file_path.exists():
    st.write(f"Running {selected_hw}…")
    try:
        exec(open(file_path).read(), globals())
    except ModuleNotFoundError as e:
        st.error(f"Missing dependency: {e.name}. Install with `pip install {e.name}`")
    except Exception as e:
        st.error(f"Error running {selected_hw}: {e}")
else:
    st.error(f"{file_path} not found.")
