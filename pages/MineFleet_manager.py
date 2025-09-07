import pandas as pd
import numpy as np
import streamlit as st
import pickle
import os

from func.log import log

log()

"# :material/conveyor_belt: Mine fleet manager"
"""
Here you can specify machine number, availability and capacity.
"""

# select a minops
list_of_minops  = [ x for x in os.listdir(st.session_state.project) if x.startswith("MineOps - ") and x.endswith(".pkl") ]
selected_minops = st.selectbox("Select a MineOps", list_of_minops, format_func = lambda x: x.replace(".pkl",""))

list_module = [":green-badge[:material/add_circle:] Create a new MineFleet", ":orange-badge[:material/edit_square:] Modify an existing MineFleet"]
selected_module = st.pills("Select what you want to do :", list_module, selection_mode="single",
                           label_visibility="hidden", default=list_module[0])

if selected_minops is None or selected_minops == "" :
  st.info("No MineOps selected, you have to create a MineOps Class first")
  st.stop()
fpath           = os.path.join( st.session_state.project, selected_minops )

# read minops
@st.cache_data
def read_minops(fpath) :
  with open(fpath, "rb") as f :
    mo = pickle.load(f)
  return mo
minops = read_minops(fpath)

if len(list(minops.mine_fleet))>0 and selected_module == list_module[0] :
  st.info("You are about to overwrite existing MineFleet for your MineOps. A MineOps class can only have one MineFleet.")

if selected_module == list_module[0] :
  minops.create_fleet()
else :
  ""

