"""lrl-toolkit web dashboard (Streamlit).

M0 status: a minimal shell showing the four planned tabs. M5 wires these up:
  1. Wizard        — pick language/base model/compute -> generate a project.yaml
  2. Run & monitor — trigger stages, stream logs, view data/model cards
  3. Review        — human-in-the-loop conversational-pair review queue
  4. Chat          — talk to the exported model

Run with: ``lrl dashboard`` (or ``streamlit run dashboard/app.py``).
"""

from __future__ import annotations

import streamlit as st

from lrl_toolkit import __version__
from lrl_toolkit.config import STAGE_ORDER
from lrl_toolkit.registry import list_compute, list_languages, list_models

st.set_page_config(page_title="lrl-toolkit", page_icon="🗣️", layout="wide")
st.title("🗣️ lrl-toolkit")
st.caption(f"Build custom LLMs for low-resource languages · v{__version__}")

wizard, monitor, review, chat = st.tabs(["Wizard", "Run & monitor", "Review", "Chat"])

with wizard:
    st.subheader("New project")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.selectbox("Language", list_languages() or ["(add a profile)"])
    with col2:
        st.selectbox("Base model", list_models() or ["(add a profile)"])
    with col3:
        st.selectbox("Compute", list_compute() or ["(add a profile)"])
    st.info("M5: this tab will generate a `project.yaml` and hand off to the runner.")

with monitor:
    st.subheader("Pipeline")
    st.write("Stages: " + " → ".join(STAGE_ORDER))
    st.info("M5: trigger stages, stream logs, and view data/model cards here.")

with review:
    st.subheader("Conversational-pair review")
    st.info("M5: native-speaker review queue (accept / edit / reject) feeds the SFT set.")

with chat:
    st.subheader("Chat with your model")
    st.info("M5: load the exported GGUF and chat with it here.")
