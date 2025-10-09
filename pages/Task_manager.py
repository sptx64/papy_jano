import pandas as pd
import numpy as np
import os
import streamlit as st
from func.log import log
from func.all_class import MineTask
import random
import pickle


from streamlit_echarts import st_echarts
import plotly.graph_objects as go

log()

"# :material/assignment_add: Project manager"

"Here you can setup your task and schedule your activities"

# select a minops
list_of_minops  = [ x for x in os.listdir(st.session_state.project) if x.startswith("MineOps - ") and x.endswith(".pkl") ]
selected_minops = st.selectbox("Select a MineOps", list_of_minops, format_func = lambda x: x.replace(".pkl",""))

list_module = [":green[:material/add_circle:] Create a new TaskSet", ":orange[:material/edit_square:] Modify an existing TaskSet"]
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
    
    with c[i%ncol].popover(f"Task **{i}** {" - " + str(task_dict['Task name']) if task_dict['Task name'] is not None else ''}",
                           icon=":material/assignment_add:", width="stretch") :
      
      with st.form(id, border=False) :
        t = st.tabs(["General", "Machines","Delays", "Other"])
        task_dict["Task category"] = t[0].selectbox(f"category:", list_tasks, )
        task_dict["Supervisor"]    = t[0].selectbox(f"supervisor", list_supervisors)
        task_dict["Progress"]      = t[0].slider("Progress", 0, 100, 0, 1)
        task_dict["Machines"]      = {}
        for mt in list_machines :
          task_dict["Machines"][mt] = t[1].number_input(f"Required {mt}", 0, 1000, 0)

        col1,col2 = t[2].columns(2)
        task_dict["Start date"] = col1.date_input(f"Start date", "today")
        task_dict["Start date known"] = col1.toggle(f"Start date is known", value=False)
        if task_dict["Start date known"] :
          task_dict["Start date"] = None
        
        task_dict["End date"] = col2.date_input(f"End date", "today")
        task_dict["End date known"] = col2.toggle(f"End date is known", value=False)
        if task_dict["End date known"] :
          task_dict["End date"] = None
        
        task_dict["delay_optimistic"] = col1.number_input("Optimistic delay to complete (days)",1,9999,1)
        task_dict["delay_probable"] = col2.number_input("Probable delay to complete (days)",1,9999,1)
        task_dict["delay_pessimistic"] = col1.number_input("Pessimistic delay to complete (days)",1,9999,1)
        task_dict["lag"] = col2.number_input("Lag (days)",0,9999,0)

        task_dict["dependencies"]  = t[3].multiselect(f"Enter the Task ID dependencies", [], [], accept_new_options=True, help="integers linking to the parent task IDs (0,1,2,3,...)")
        task_dict["dependency_type"]  = t[3].selectbox(f"Dependency type", ["FS", "SS", "FF", "SF"])
        task_dict["comments"] = t[3].text_area(f"Comments", None)

        st.form_submit_button("Submit", type="primary", width="stretch")
    
    save_dict[i] = task_dict

    machine_text, nb_machine_text = [ k for k in task_dict["Machines"] ], [ str(task_dict["Machines"][k]) for k in task_dict["Machines"] ]
    machine_text = [ f":red[{x[:1]}]:red-badge[{y}]" for x,y in zip(machine_text, nb_machine_text) ]
    machine_text = " ".join(machine_text)
    
    
    text_param = f"""
    :blue-badge[{task_dict['Supervisor']}]
    :green-badge[{task_dict['Start date']}] :orange-badge[{task_dict['End date']}] 

    {machine_text} :orange-badge[{' '.join(task_dict['dependencies'])}]
    :violet-badge[{task_dict['delay_optimistic']}] 
    :violet-badge[{task_dict['delay_probable']}] 
    :violet-badge[{task_dict['delay_pessimistic']}]
    
    """
    
    c[i%ncol].write(text_param)
    c[i%ncol].progress(task_dict['Progress']/100)
    if i%ncol == ncol-1 :
      c = st.columns(ncol, border=True)

  c = st.columns(6)
  if c[0].button("Show dependencies") :
    task_coords = {}
    val = 1; i=0
    for k in save_dict :
      task_coords[f"Task {k}"] = [random.uniform(0,4), random.uniform(0,4)]
      

    
    
    links=[]
    for k in save_dict :
      if len(save_dict[k]["dependencies"]) == 0 :
        task_coords[f"Task {k}"][1]=5
        
    for k in save_dict :
      if len(save_dict[k]["dependencies"]) > 0 :
        for d in save_dict[k]["dependencies"] :
          links.append({"x":[task_coords[f"Task {d}"][0], task_coords[f"Task {k}"][0]],  "y": [task_coords[f"Task {d}"][1], task_coords[f"Task {k}"][1]]})


    fig = go.Figure()
    for k in task_coords :
      fig.add_trace(go.Scatter(x=[task_coords[k][0]], y=[task_coords[k][1]], mode="markers+text", text=[k], textposition="top center", marker_size=20, name=k, showlegend=True))
    for l in links :
      fig.add_trace(go.Scatter(x=l["x"], y=l["y"], mode="markers+lines", marker= dict(size=5,symbol= "arrow-bar-up", angleref="previous"), line_width=1, line_dash="dash", line_color="black"))

    fig.update_layout(template="simple_white")
    st.plotly_chart(fig)
      
  





               




  if c[1].button("Save", type="primary") :
    for k in save_dict :
      if save_dict[k]["Task name"] is None :
        st.toast("A task name is None", icon=":material/warning:")
      msum=0
      for j in save_dict[k]["Machines"] :
        msum+=save_dict[k]["Machines"][j] if j is not None else 0
      if msum == 0 :
        st.toast(f":material/warning: Task{k}, {save_dict[k]['Task name']} Number of required machines is equal to 0", icon=":material/warning:")


    dict_all_task = {}
    for k in save_dict :
      task = MineTask(ID               = k,
                  name                 = save_dict[k]['Task name'],
                  category             = save_dict[k]['Task category'],
                  supervisor           = save_dict[k]['Supervisor'],
                  required_machines    = save_dict[k]["Machines"],
                  progress             = save_dict[k]["Progress"],
                  start_date           = save_dict[k]["Start date"],
                  end_date             = save_dict[k]["End date"],
                  duration_optimistic  = save_dict[k]['delay_optimistic'],
                  duration_pessimistic = save_dict[k]['delay_pessimistic'],
                  duration_probable    = save_dict[k]['delay_probable'],
                  lag                  = save_dict[k]['lag'],
                  dependencies         = save_dict[k]['dependencies'],
                  dependency_type      = save_dict[k]['dependency_type'],
                  comments             = save_dict[k]["comments"],
                 )
      dict_all_task[k] = task
    minops.mine_task = dict_all_task
    minops.save_pkl()

elif selected_module == list_module[1] :
  "_Coming soon..._"


      
        

  # if st.button("Save", type="primary"):
  #   for k in save_dict :
  #     class_task = Task()
    


    
    

  

