import pandas as pd
import numpy as np
import os
import streamlit as st
from func.log import log
from func.all_class import Task
import pickle

log()

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
  ncol=3
  c = st.columns(ncol)
  task_num = c[0].number_input("How many tasks are you scheduling?", 1, 50, 4)
  save_dict = {}
  c = st.columns(ncol, border=True)
  for i in range(task_num) :
    
    id, task_dict          = f"Task {i}", {}
    task_dict["Task name"] = c[i%ncol].text_input(f"Task **{i}** name:", None, placeholder="YOUR TASK")
    
    with c[i%ncol].popover(f"Task **{i}** {" - " + str(task_dict['Task name']) if task_dict['Task name'] is not None else ''}", width="stretch") :
      
      with st.form(id, border=False) :
        t = st.tabs(["General", "Machines","Delays", "Other"])
        task_dict["Task category"] = t[0].selectbox(f"category:", list_tasks, )
        task_dict["Supervisor"]    = t[0].selectbox(f"supervisor", list_supervisors)
        task_dict["Progress"]      = t[0].number_input("Progress", 0, 100, 0)
        task_dict["Machines"]      = {}
        for mt in list_machines :
          task_dict["Machines"][mt] = t[1].number_input(f"Required {mt}", 0, 1000, 0)

        task_dict["Start date"] = t[2].date_input(f"Start date", "today")
        task_dict["Start date known"] = t[2].toggle(f"Start date is known", value=False)
        if task_dict["Start date known"] :
          task_dict["Start date"] = None
        
        task_dict["End date"] = t[2].date_input(f"End date", "today")
        task_dict["End date known"] = t[2].toggle(f"End date is known", value=False)
        if task_dict["End date known"] :
          task_dict["End date"] = None
        
        task_dict["Delay_optimistic"] = t[2].number_input("Optimistic delay to complete (days)",1,9999,1)
        task_dict["Delay_probable"] = t[2].number_input("Probable delay to complete (days)",1,9999,1)
        task_dict["Delay_pessimistic"] = t[2].number_input("Pessimistic delay to complete (days)",1,9999,1)
        task_dict["lag"] = t[2].number_input("Lag (days)",0,9999,0)

        task_dict["Dependencies"]  = t[3].multiselect(f"Enter the Task ID dependencies", [], [], accept_new_options=True)
        task_dict["Dependency type"]  = t[3].selectbox(f"Dependency type", ["FS", "SS", "FF", "SF"])
        task_dict["Comments"] = t[3].text_area(f"Comments", None)

        st.form_submit_button("Submit", type="primary", width="stretch")
    
    save_dict[i] = task_dict

    machine_text, nb_machine_text = [ k for k in task_dict["Machines"] ], [ str(task_dict["Machines"][k]) for k in task_dict["Machines"] ]
    machine_text = [ f":red[{x[:1]}]:red-badge[{y}]" for x,y in zip(machine_text, nb_machine_text) ]
    machine_text = " ".join(machine_text)
    
    
    text_param = f":blue-badge[{task_dict['Supervisor']}]  {machine_text} :orange-badge[{' '.join(task_dict['Dependencies'])}] :green-badge[{task_dict['Start date']}] :orange[{task_dict['End date']}]"
    c[i%ncol].write(text_param)
    if i%ncol == ncol-1 :
      c = st.columns(ncol, border=True)


  with st.expander(":material/warning: Warnings", expanded=True) :
    st.write(save_dict)
    for k in save_dict :
      msum=0
      for j in save_dict[k]["Machines"] :
        msum+=save_dict[k]["Machines"][j] if j is not None else 0
      if msum == 0 :
        st.warning(f":material/warning: Task{k}, {save_dict[k]['Task name']} Number of required machines is equal to 0")

  # if st.button("Save", type="primary"):
  #   for k in save_dict :
  #     class_task = Task()
    


    
    

  

