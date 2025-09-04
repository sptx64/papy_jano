import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import time

from func.data import load_opt_pkl

"# List manager"
"""
Here you can manage your options for the app modules such as task, supervisors and mine fleet available machines type. This is done to simplify the next modules éditions with more sélection than typing. Also it's désigned to design your full mining fleet once only.
"""

#loading the existing pkl if there are any
fname_begin, folder = "dict_opt", "files"
load_opt_pkl(start_with=fname_begin, location=folder)

#Tasks
if "Task" not in st.session_state.dict_opt :
  st.session_state.dict_opt["Task"] = []
list_task = st.multiselect(":material/assignment_add: Task options that will be available in other modules", st.session_state.dict_opt["Task"], st.session_state.dict_opt["Task"], accept_new_options=True)

#Supervisors
if "Supervisors" not in st.session_state.dict_opt :
  st.session_state.dict_opt["Supervisors"] = []
list_supervisors = st.multiselect(":material/man: Supervisors options that will be available in other modules", st.session_state.dict_opt["Supervisors"], st.session_state.dict_opt["Supervisors"], accept_new_options=True)

#Machines
if "Machines" not in st.session_state.dict_opt :
  st.session_state.dict_opt["Machines"] = []
list_machines = st.multiselect(":material/conveyor_belt: Machines that will be available in other modules", st.session_state.dict_opt["Machines"], st.session_state.dict_opt["Machines"], accept_new_options=True)

#one unique save button that will save all lists in a dictionary and deleting any duplicate option
if st.button("Save options", type="primary", help="This button save all options (Task + Supervisors + Machines)  in one click.") :
  dict_opt = {
    "Task" : sorted(list(np.unique(list_task)), key=str.lower),
    "Supervisors" : sorted(list(np.unique(list_supervisors)), key=str.lower),
    "Machines" : sorted(list(np.unique(list_machines)), key=str.lower),
  }
  #saving in the cache of streamlit (st.session_state)
  st.session_state.dict_opt = dict_opt

  #storing it locally for next sessions
  full_name = os.path.join(folder, f"{fname_begin}.pkl")
  with open(full_name, "wb") as f :
    pickle.dump(dict_opt, f)
    st.toast("The options have been saved!", icon=":material/check_small:")
    st.success("Saved! The app will rerun in 3 seconds.")
    time.sleep(3)
    st.rerun()
    
    
    

  

  


  
