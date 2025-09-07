import os
import numpy
import streamlit as st
#from .data import get_save_folder

def get_save_folder() :
  return "files"

def log() :
  if "project" in st.session_state :
    st.sidebar.caption(f"_Project : {st.session_state.project_name.upper()}_")
    if st.sidebar.button("Log out", width="content", type="primary") :
      for k in st.session_state :
        del st.session_state[k]
      st.cache_data.clear()
      st.rerun()

  #if project not in st.session_state :
  else :
    
    @st.dialog("Log in", dismissible=False)
    def log_modal() :
      ffolder = get_save_folder()
    
      projects = [ n for n in os.listdir(ffolder) if os.path.isdir(os.path.join(ffolder,n)) ]
      
      c = st.columns(2, vertical_alignment="bottom")
      disabled=True
      new_project = c[1].toggle("Create a New project", value=True)
      if new_project :
        project=c[0].text_input(":green-badge[:material/add_circle:] New project name", None, placeholder="YOUR NEW PROJECT NAME")
        if project in projects :
          st.warning("This project name already exists")
        elif project is None :
          st.info("Enter a new project name")
        else :
          disabled=False

      else :
        project=c[0].selectbox(":orange-badge[:material/edit_square:] Select your existing project", projects, None)
        if project is not None :
          disabled=False
       
      if st.button("Log in", type="primary", disabled=disabled) :
        filepath = os.path.join(ffolder,project)
        if new_project :
          os.mkdir(filepath)
      
        st.session_state.project=filepath
        st.session_state.project_name=project
        st.rerun()
      else :
        st.stop()
    log_modal()
        

    
