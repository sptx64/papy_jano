import pandas as pd
import numpy as np
import streamlit as st
from streamlit_echarts import st_echarts
from func.simulation import _triangular_safe, _beta_pert_safe, _sample_activity

import os
from func.log import log
from func.all_class import MineTask
import pickle

from datetime import date

log()

"# :material/assignment_add: Project manager"

"Here you can setup your task and schedule your activities"

# select a minops
list_of_minops  = [ x for x in os.listdir(st.session_state.project) if x.startswith("MineOps - ") and x.endswith(".pkl") ]
selected_minops = st.selectbox("Select a MineOps", list_of_minops, format_func = lambda x: x.replace(".pkl",""))

if selected_minops is None or selected_minops == "" :
  st.info("No MineOps selected, you have to create a MineOps Class first")
  st.stop()
fpath = os.path.join( st.session_state.project, selected_minops )

# read minops
def read_minops(fpath) :
  with open(fpath, "rb") as f :
    mo = pickle.load(f)
  return mo
minops = read_minops(fpath)

mine_task = minops.mine_task

if mine_task is None :
  st.info("Create and manage your tasks in the Task manager. Don't forget to click save.")
  st.stop()


coefficient_non_critical = st.sidebar.number_input("Non-critical coefficient", value=0.5, min_value=0.1, max_value=2.0, step=0.1, help="Used to compute the buffer between non-critical tasks")
coefficient_critical = st.sidebar.number_input("Critical coefficient", value=1.3, min_value=0.1, max_value=2.0, step=0.1, help="Used to compute the buffer between critical tasks")
multiplier_multi_dependencies = st.sidebar.number_input("Multi-dependencies multiplier", value=1.2, min_value=1.0, max_value=3.0, step=0.1,help="Additional multiplier for multi-dependency tasks")


type_distrib = st.sidebar.selectbox("Distrib. method", ["Triangular","Beta-PERT"])

sim_fleet_avail=st.sidebar.toggle("Simulate fleet availability")
if sim_fleet_avail :
  sim_break = st.toggle("Simulate machine breakdown")
  break_down_delay = st.slider("Break down delay",0,50,(0,10),1, help="a random number in this range will be picked to simulate the machine hazardous breakdown" ) if sim_break else None
else :
  sim_break=False;break_down_delay=None
  
start_full_fleet=st.sidebar.toggle("Start a new task only when full fleet is available", value=True, disabled=True)



t = st.tabs(["Dashboard","Progress"])
with t[0] :
  "## Dashboard"
  c = st.columns([1,2], vertical_alignment="center")
  c[0].metric("Total tasks", value=len([ x for x in mine_task ]), border=True )
  
  today = date.today()
  st.write(today)
  c[0].metric("Running tasks", value=len([ x for x in mine_task if mine_task[x].start_date is not None ]), border=True )
  
  
  plot = c[1].pills("Plot type", ["category","supervisor"], default="category", selection_mode="single")
  if plot is not None :
    if plot == "category" :
      res = [ [1, k, mine_task[k].name, mine_task[k].category] for k in mine_task ]
    elif plot == "supervisor" :
      res = [ [1, k, mine_task[k].name, mine_task[k].supervisor] for k in mine_task ]

    df = pd.DataFrame(res, columns=["count", "ID", "name",plot])
    df_grp = df[[plot,"count"]].groupby(plot).sum().reset_index()
  
    res = [ {"value" : float(v), "name":n } for n,v in zip(df_grp[plot].values, df_grp["count"].values) ]
    options = {
      "title" : {"text" : 'Tasks', "subtext" : f'Tasks by {plot}', "left" : 'center'},
      "tooltip" : { "trigger" : 'item'},
      "legend"  : {"orient" : 'vertical', "left" : 'left'},
      "series"  : [
            { "name" : 'Access From', "type"   : 'pie', "radius" : '70%',
              "data" : res,
              "emphasis" : {
                "itemStyle" : { "shadowBlur" : 10, "shadowOffsetX" : 0, "shadowColor" : 'rgba(0, 0, 0, 0.5)' }
              }
            }
          ]
    }
    
    with c[1] :
      st_echarts(options=options, height="600px",)

with t[1] :
  c = st.columns([1,2])
  res = [ ["Task name", "Task category", "Progress"] ]
  for k in mine_task :
    mine_task[k].progress = float( c[0].slider(f"{k}-{mine_task[k].name} progress", 0, 100, mine_task[k].progress) )
    res.append([mine_task[k].name, mine_task[k].category, mine_task[k].progress])
    
options = {
  "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
  "dataset" : { "source" : res },
  "grid"    : { "containLabel" : True },
  "xAxis"   : { "name" : "Progress" },
  "yAxis"   : { "type" : "category" },
  "visualMap" : {
    "orient"    : "horizontal",
    "left"      : "center",
    "min"       : 0.,
    "max"       : 100.,
    "text"      : ["High Progress", "Low Progress"],
    "dimension" : 2,
    "inRange"   : { "color" : ["#FD665F", "#FFCE34", "#65B581"] }
  },
  "series" : [{ "type" : "bar", "encode" : { "x" : "Progress", "y" : "Task name" } }]
}


with c[1] :
  px = 80 * len(res)
  st_echarts(options=options, height=f"{px}px",)

mine_fleet = minops.mine_fleet

mine_task

if st.button("Go") :
  ""
  # mine_fleet = minops.mine_fleet
  
  # indep_tasks = [k for k in mine_task if len(mine_task[k].dependencies) == 0 ]
  # dep_tasks   = [k for k in mine_task if len(mine_task[k].dependencies) > 0  ]

  # done = []; active = [];
  # unlocked_tasks = indep_tasks.copy(); locked_tasks = dep_tasks.copy();
  # remaining = indep_tasks + dep_tasks
  
  # quit=360; day=1;
  # while (len(remaining) > 0) and day<quit :
  #   available_fleet = { k:len([x for x in mine_fleet[k] if mine_fleet[k][int(x)].availability == True]) for k in mine_fleet }
  #   random.shuffle(unlocked_tasks)
  #   for ut in unlocked_tasks :
  #     rm = mine_task[ut].required_machines
  #     delta = [ available_fleet[k] - rm[k] for k in rm ]
  #     if all(x >= 0 for x in delta) :
  #       slct_machines = { k:random.sample([ mt for mt in mine_fleet[k] ], rm[k]) k for k in rm }
  #       active.append({ut:{"mine_task":mine_task[ut], "locked_machines":slct_machines}, "progress":mine_task[ut].progress, "duration":"" })
        
      
      
    # day += 1
    
    
  
