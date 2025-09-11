import streamlit as st
import pandas as pd
import numpy as np



st.set_page_config(layout="wide")
pages_dict = {
  "HOME"   : [st.Page("pages/Home.py", title="Home", icon=":material/elderly:")],
  "SETUP" : [
    st.Page("pages/Import.py", title="Import files", icon=":material/upload:"),
    st.Page("pages/MineOps_manager.py", title="MineOps manager", icon=":material/event_list:"),
    st.Page("pages/MineFleet_manager.py", title="Mine Fleet manager", icon=":material/front_loader:"),
             ],
  "MINE PLAN" : [
    st.Page("pages/Task_manager.py", title="Task manager", icon=":material/assignment_add:"),
                ],
  "APP DE BOLOSS" : [
    st.Page("pages/app.py", title="vieille appli dégueux", icon=":material/delete:"),
    st.Page("pages/visualizations.py", title="visualisations pas ouf", icon=":material/earthquake:"),
  ]
}

pg = st.navigation( pages_dict, position="top" )

pg.run()
