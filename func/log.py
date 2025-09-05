import os
import numpy
import streamlit as st
from data import get_save_folder


def log() :
  if st.sidebar.button("Log out", use_container_width=True, type="primary") :
    for k in st.session_state :
      del st.session_state[k]
    st.cache_data.clear()
    st.rerun()


  if not "project" in st.session_state :
    ffolder = get_save_folder()
    
    projects = [ n for n in os.listdir(ffolder) if os.path.isdir(os.path.join(ffolder,n)) ]
    "## Log to your project"
    c = st.colums(2)
    disabled=True
    new_project = c[1].toggle("Create a New project", value=True)
    if new_project :
      project=c[0].text_input("New project name", None, placeholder="YOUR NEW PROJECT NAME")
      if project in projects or project is None :
        st.warning("This project name is empty or already exists")
      else :
        disabled=False

    else :
      project=st.selectbox("Select your existing project", projects, None)
      if project is not None :
        disabled=False
       
    if st.button("Log in", type="primary", disabled=disabled) :
      filepath = os.path.join(ffolder,project)
      if new_project :
        os.mkdir(filepath)
      
      st.session_state.project=filepath
      st.rerun()
      

    