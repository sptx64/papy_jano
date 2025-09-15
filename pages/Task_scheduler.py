import pandas as pd
import numpy as np
import streamlit as st
from streamlit_echarts import st_echarts

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


t = st.tabs(["Dashboard","Progress"])
with t[0] :
  "## Dashboard"
  c = st.columns([1,2])
  c[0].metric("Total tasks", value=len([ x for x in mine_task ]), border=True )
  
  today = date.today()
  st.write(today)
  c[0].metric("Running tasks", value=len([ x for x in mine_task if mine_task[x].start_date is not None ]), border=True )
  
  
  
  # task by category
  res = [ [1, k, mine_task[k].name, mine_task[k].category] for k in mine_task ]
  df = pd.DataFrame(res, columns=["count", "ID", "name","category"])
  df_grp = df[["category","count"]].groupby("category").sum().reset_index()

  res = [ {"value" : float(v), "name":n } for n,v in zip(df_grp["category"].values, df_grp["count"].values) ]
  options = {
    "title" : {"text" : 'Tasks', "subtext" : 'Tasks by category', "left" : 'center'},
    "tooltip" : { "trigger" : 'item'},
    "legend"  : {"orient" : 'vertical', "left" : 'left'},
    "series"  : [
          { "name" : 'Access From', "type"   : 'pie', "radius" : '50%',
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
    mine_task[k].progress = float( c[0].slider(f"{k} - Progress", 0, 100, mine_task[k].progress) )
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
    "inRange"   : { "color" : ["#65B581", "#FFCE34", "#FD665F"] }
  },
  "series" : [{ "type" : "bar", "encode" : { "x" : "Progress", "y" : "Task name" } }]
}

# options = {
#   "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
#   "xAxis": {
#     "type": "category",
#     "data": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
#   },
#   "yAxis": {"type": "value"},
#   "series": [{"data": [120, 200, 150, 80, 70, 110, 130], "type": "bar"}],
# }




with c[1] :
  px = 80 * len(res)
  st_echarts(options=options, height=f"{px}px",)
