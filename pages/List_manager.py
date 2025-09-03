import streamlit as st
import pandas as pd
import numpy as np
import pickle

"# List manager"
"On this page you can manage all options that you want to be added or removed to lists."

def load_opt_pkl(start_with, location) :
  if "dict_opt" not in st.session_state :
    #checking if the folder in input exists
    if not os.path.exists(location) :
      st.error(f":material/error: There was an error : the folder {location} does not exist")
      st.stop()
  
    #storing a list of all files that start with the key start_with
    list_files = [ os.path.join(location,x) for x in os.listdir(location) if x.startswith(start_with) and x.endswith(".pkl")]
    if len(list_files) >= 1 :
      # if there are files that starts with start_with and ends with pkl
      with open(list_files[0], rb) as f :
        st.session_state.dict_opt = pickle.load(f)
      
      if len(list_files) == 1 :
        st.toast(f":green-badge[Great!] the file {list_files[0]} have been imported successfully", icon=":material/check_small:")
      else :
        st.toast(f":orange-badge[Warning!] there was more than one file in the folder {location} with a name starting with {start_with}. The first file encoutered : {list_file[0]} has been loaded by default.", icon=":material/warning:")
        
    else :
      st.session_state.dict_opt = {}    
    

list_opt_id, folder = "dict_opt", "files"
load_opt_pkl(start_with=list_opt_id, location=folder)

t = st.tabs(["Task", "Task supervisor", "Machines"])
with t[0] :
  if "Task" not in st.session_state.dict_opt :
    st.session_state.dict_opt["Task"] = []

with t[1] :
  if "Task supervisor" not in st.session_state.dict_opt :
    st.session_state.dict_opt["Task supervisor"] = []

with t[2] :
  if "Machines" not in st.session_state.dict_opt :
    st.session_state.dict_opt["Machines"] = []



  
