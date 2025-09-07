import pandas as pd
import numpy as np
import streamlit as st
import pickle
import os

"# :material/conveyor_belt: Mine fleet manager"
"""
Here you can specify machine number, availability and capacity.
"""

# select a minops
st.session_state.project
list_of_minops  = [ x for x in os.listdir(st.session_state.project) if x.startswith("MineOps - ") and x.endswith(".pkl") ]
selected_minops = st.selectbox("Select a MineOps", list_of_minops, format_func = lambda x: x.replace(".pkl",""))
fpath           = os.path.join( st.session_state.project, selected_minops )

# read minops
@st.cache_data
def read_minops(fpath) :
  with open(fpath, "rb") as f :
    mo = pickle.load(f)
  return mo

minops = read_minops(fpath)


st.cache_data.clear()
