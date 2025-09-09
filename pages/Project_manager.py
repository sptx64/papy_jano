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
    with c[i%ncol].popover(f"Task {ncol}", width="stretch") :
      id = f"Task {i}"
      with st.form(id) :
        task_dict={}
        task_dict["Task category"] = st.selectbox(f"Task {ncol} category:", list_tasks, )
        task_dict["Task name"]     = st.text_input(f"Task {ncol} name", None, placeholder="YOUR TASK")
        task_dict["Supervisor"]    = st.selectbox(f"Task {ncol} supervisor", list_supervisors)
        task_dict["Machines"]      = {}
        for mt in list_machines :
          task_dict["Machines"][mt] = st.number_input(f"Task {ncol} required {mt}", 0, 1000, 0)

        task_dict["Dependencies"]  = st.multiselect(f"Task {ncol} Enter the Task ID dependencies", [], [])
        task_dict["Start date"] = st.date_input(f"Task {ncol} Start date", "today")
        task_dict["Comments"] = st.text_input(f"Task {ncol} Comments", None)
        st.form_submit_button("Submit")





  













  
