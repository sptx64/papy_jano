import pandas as pd
import numpy as np
import os
import streamlit as st
from func.log import log

"# :material/assignment_add: Project manager"

"Here you can setup your tasks and schedule your activities"

# select a minops
list_of_minops  = [ x for x in os.listdir(st.session_state.project) if x.startswith("MineOps - ") and x.endswith(".pkl") ]
selected_minops = st.selectbox("Select a MineOps", list_of_minops, format_func = lambda x: x.replace(".pkl",""))

list_module = [":green-badge[:material/add_circle:] Create a new TaskSet", ":orange-badge[:material/edit_square:] Modify an existing TaskSet"]
selected_module = st.pills("Select what you want to do :", list_module, selection_mode="single",
                           label_visibility="hidden", default=list_module[0])

if selected_minops is None or selected_minops == "" :
  st.info("No MineOps selected, you have to create a MineOps Class first")
  st.stop()
fpath = os.path.join( st.session_state.project, selected_minops )

# read minops
@st.cache_data
def read_minops(fpath) :
  with open(fpath, "rb") as f :
    mo = pickle.load(f)
  return mo
minops = read_minops(fpath)
list_tasks = minops.dict_opt["Task"]
list_supervisors = minops.dict_opt["Supervisors"]
list_machines = minops.dict_opt["Machines"]


#Create a new task set
if selected_module == list_module[0] :
  task_num = st.number_input("How many tasks are you scheduling?", 1, 50, 4)
  ncol=4
  save_dict = {}
  c = st.columns(ncol)
  for i in range(task_num) :
    
    id, task_dict          = f"Task {i}", {}
    task_dict["Task name"] = c[i%ncol].text_input(f"Task {i} name:", None, placeholder="YOUR TASK")
    
    with c[i%ncol].popover(f"{id}-{task_dict['Task name']}", width="stretch") :
      
      with st.form(id) :
        t = st.tabs(["General", "Machines", "Other"])
        task_dict["Task category"] = t[0].selectbox(f"Task {i} category:", list_tasks, )
        task_dict["Supervisor"]    = t[0].selectbox(f"Task {i} supervisor", list_supervisors)
        task_dict["Machines"]      = {}
        for mt in list_machines :
          task_dict["Machines"][mt] = t[1].number_input(f"Task {i} required {mt}", 0, 1000, 0)

        task_dict["Dependencies"]  = t[2].multiselect(f"Task {i} Enter the Task ID dependencies", [], [])
        task_dict["Start date"] = t[2].date_input(f"Task {i} Start date", "today")
        task_dict["Comments"] = t[2].text_input(f"Task {i} Comments", None)
        st.form_submit_button("Submit")





  













  
