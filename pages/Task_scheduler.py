import pandas as pd
import numpy as np
import streamlit as st
from streamlit_echarts import st_echarts

import os
from func.log import log
from func.all_class import MineTask
from func.data import TaskManager
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


coefficient_non_critical = st.sidebar.number_input("Non-critical coefficient", value=0.5,
                                                   min_value=0.1, max_value=2.0, step=0.1,
                                                   help="Used to compute the buffer between non-critical tasks")

coefficient_critical = st.sidebar.number_input("Critical coefficient", value=1.3, min_value=0.1, 
                                               max_value=2.0, step=0.1, help="Used to compute the buffer between critical tasks")

multiplier_multi_dependencies = st.sidebar.number_input("Multi-dependencies multiplier", value=1.2, min_value=1.0, 
                                                        max_value=3.0, step=0.1,
                                                        help="Additional multiplier for multi-dependency tasks")



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


if st.button("Go") :
  if "task_manager" not in st.session_state:
    st.session_state.task_manager = TaskManager()

  tm = st.session_state.task_manager
  
  # === CALCULS DES COEFFS ===
  coeff_non   = round(coefficient_non_critical, 2)
  coeff_crit  = round(coefficient_critical, 2)
  coeff_multi = round(multiplier_multi_dependencies, 2)

  #################
  #Synchroniser les tasks
  #################
  
  tm.save()
  # === DATA VALIDATION DISPLAY ===
  errors = tm.validate_task_data()
  if errors:
    st.sidebar.error("⚠️ Erreurs détectées:")
    for error in errors:
      st.sidebar.write(f"• {error}")

  st.write(f"**Total:** {len(tm.tasks)} tâches")
  # === DATA EDITOR SETUP ===
  # Convert tasks to DataFrame format for editing
  records = []
  for task in sorted(tm.tasks.values(), key=lambda x: x.id):
    # General Information
    records.append({
            "id": task.id,
            "name": task.name,
            "category": task.category,
            "responsible": task.responsible,
            "equipment": task.equipment,
            "comments": task.comments,
            "dependencies": task.dependencies,
            # Scheduling
            "start_date": task.start_date.isoformat() if task.start_date else "",
            "end_date": task.end_date.isoformat() if task.end_date else "",
            "lag": task.lag,
            "progress": task.progress,
            "projected_end_date": task.projected_end_date.isoformat() if task.projected_end_date else "",
            "projection_speed": round(task.projection_speed, 2) if task.projection_speed is not None else "",
            "is_critical": task.is_critical,
            # Durations
            "duration_optimistic": task.duration_optimistic,
            "duration_pessimistic": task.duration_pessimistic,
            "duration_probable": task.duration_probable,
            "duration_stochastic": round(task.duration_stochastic, 2) if task.duration_stochastic is not None else "",
            "duration_days": task.duration_days,
            # Risk & Uncertainty
            "standard_deviation": round(task.standard_deviation, 2) if task.standard_deviation is not None else "",
            "buffer": round(task.buffer, 2) if task.buffer is not None else "",
            "p10": round(task.p10, 2) if task.p10 is not None else "",
            "p20": round(task.p20, 2) if task.p20 is not None else "",
            "p30": round(task.p30, 2) if task.p30 is not None else "",
            "p40": round(task.p40, 2) if task.p40 is not None else "",
            "p50": round(task.p50, 2) if task.p50 is not None else "",
            "p60": round(task.p60, 2) if task.p60 is not None else "",
            "p70": round(task.p70, 2) if task.p70 is not None else "",
            "p80": round(task.p80, 2) if task.p80 is not None else "",
            "p90": round(task.p90, 2) if task.p90 is not None else "",
        })


    
  # Add empty row if no tasks exist
  if not records:
    records.append({
            "id": "",
            "name": "",
            "category": "Task",
            "responsible": "",
            "equipment": "",
            "comments": "",
            "dependencies": "",
            "start_date": "",
            "end_date": "",
            "lag": 0,
            "progress": 0,
            "projected_end_date": "",
            "projection_speed": "",
            "is_critical": None,
            "duration_optimistic": None,
            "duration_pessimistic": None,
            "duration_probable": None,
            "duration_stochastic": "",
            "duration_days": None,
            "standard_deviation": "",
            "buffer": "",
            "p10": "", "p20": "", "p30": "", "p40": "", "p50": "",
            "p60": "", "p70": "", "p80": "", "p90": "",
        })

    
  # === COLUMN CONFIGURATION ===
  # Match the Task dataclass exactly: Use field names as keys, emojis as headers for clarity.
  # Disabled=True for calculated fields.
  column_config = {
      # General Information
        "id": st.column_config.NumberColumn("🆔 ID", disabled=True, width=50),
        "name": st.column_config.TextColumn("📝 Name", required=True, width=150),
        "category": st.column_config.TextColumn("🏷️ Category", width=100),
        "responsible": st.column_config.TextColumn("👤 Responsible", width=120),
        "equipment": st.column_config.TextColumn("🛠️ Equipment", width=120),
        "comments": st.column_config.TextColumn("💬 Comments", width=200),
        "dependencies": st.column_config.TextColumn("🔗 Dependencies", help="Ex: 1,2,3", width=100),
        # Scheduling
        "start_date": st.column_config.TextColumn("🗓️ Start Date", help="Format: YYYY-MM-DD", width=100),
        "end_date": st.column_config.TextColumn("🎯 End Date", help="Format: YYYY-MM-DD", width=100),
        "lag": st.column_config.NumberColumn("⏳ Lag (days)", min_value=0, width=80),
        "progress": st.column_config.NumberColumn("📊 Progress (%)", min_value=0, max_value=100, step=5, width=100),
        "projected_end_date": st.column_config.TextColumn("🔮 Projected End Date", disabled=True, width=120),
        "projection_speed": st.column_config.NumberColumn("🚀 Projection Speed", disabled=True, width=100),
        "is_critical": st.column_config.CheckboxColumn("⚡ Is Critical", disabled=True, width=80),
        # Durations
        "duration_optimistic": st.column_config.NumberColumn("😊 Optimistic (days)", min_value=0, width=80),
        "duration_pessimistic": st.column_config.NumberColumn("😰 Pessimistic (days)", min_value=0, width=80),
        "duration_probable": st.column_config.NumberColumn("🤔 Probable (days)", min_value=0, width=80),
        "duration_stochastic": st.column_config.NumberColumn("📈 Stochastic (days)", disabled=True, width=80),
        "duration_days": st.column_config.NumberColumn("📅 Duration Days", min_value=0, width=80),
        # Risk & Uncertainty
        "standard_deviation": st.column_config.NumberColumn("📊 Standard Deviation", disabled=True, width=100),
        "buffer": st.column_config.NumberColumn("🛡️ Buffer (days)", disabled=True, width=80),
        "p10": st.column_config.NumberColumn("P10", disabled=True, width=60),
        "p20": st.column_config.NumberColumn("P20", disabled=True, width=60),
        "p30": st.column_config.NumberColumn("P30", disabled=True, width=60),
        "p40": st.column_config.NumberColumn("P40", disabled=True, width=60),
        "p50": st.column_config.NumberColumn("P50 (median)", disabled=True, width=60),
        "p60": st.column_config.NumberColumn("P60", disabled=True, width=60),
        "p70": st.column_config.NumberColumn("P70", disabled=True, width=60),
        "p80": st.column_config.NumberColumn("P80", disabled=True, width=60),
        "p90": st.column_config.NumberColumn("P90", disabled=True, width=60),
    }
