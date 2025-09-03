import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import time

"# List manager"
"""
On this page you can manage the lists of tasks, supervisors, and machines options which can be used across other modules.
The application enable to add, remove, or modify options for these categories through a simple interface.
List of options are cached in streamlit on this session and saved locally to persist across sessions.
"""

def load_opt_pkl(start_with, location) :
  """
  Function to load a pkl file that contain all list options
  The function return the dictionary if it's found locally on the "location" folder by looking for a file starting with "start_with"
  If a file is found then it's stored in the st.session_state
  If multiple files are found it will flag a warning for the user and the first file will be loaded
  if no file are found it will store an empty dictionary in st.session_state
  ---
  warning on the streamlit public cloud, the files stay only 24h, then are deleted.
  it will have a better behavior when run locally
  """
  
  if "dict_opt" not in st.session_state :
    #checking if the folder in input exists
    if not os.path.exists(location) :
      st.error(f":material/error: There was an error : the folder {location} does not exist")
      st.stop()
  
    #storing a list of all files that start with the key start_with
    list_files = [ os.path.join(location,x) for x in os.listdir(location) if x.startswith(start_with) and x.endswith(".pkl")]
    if len(list_files) >= 1 :
      # if there are files that starts with start_with and ends with pkl
      with open(list_files[0], "rb") as f :
        st.session_state.dict_opt = pickle.load(f)
      
      if len(list_files) == 1 :
        st.toast(f"The file {list_files[0]} have been imported successfully and cached", icon=":material/check_small:")
      else :
        st.toast(f":orange-badge[Warning!] there was more than one file in the folder {location} with a name starting with {start_with}. The first file encoutered {list_file[0]} has been loaded by default.", icon=":material/warning:")
        
    else :
      st.session_state.dict_opt = {}    
    

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
    "Task" : sorted(list(np.unique(list_task))),
    "Supervisors" : sorted(list(np.unique(list_supervisors))),
    "Machines" : sorted(list(np.unique(list_machines))),
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
    
    
    

  

  


  
