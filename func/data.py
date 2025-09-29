import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import os
import pickle

def test():
  ""

def get_save_folder() :
  return "files"

@st.cache_data
def read_minops(fpath) :
  with open(fpath, "rb") as f :
    mo = pickle.load(f)
  return mo

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


def scheduled_end(start_date, duration_days):
  if start_date and self.duration_days:
    return start_date + timedelta(days=duration_days - 1)
  else :
    return self.end_date
