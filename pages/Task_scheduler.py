import pandas as pd
import numpy as np
import streamlit as st

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

mine_task = minops.mine_task

if mine_task is None :
  st.info("Create and manage your tasks in the Task manager. Don't forget to click save.")
  st.stop()

"## Dashboard"
c = st.columns(4)
c[0].metric("Total tasks", value=len([ x for x in mine_task ]), border=True )

today = date.today()
st.write(today)
c[1].metric("Running tasks", value=len([ x for x in mine_task if mine_task[x]["Start date"] is not None ]), border=True )



# task by category
res = [ [1, k, mine_task[k].name, mine_task[k].category] for k in mine_task ]
df = pd.DataFrame(res, columns=["count", "ID", "name","category"])
df_grp = df[["category","count"]].groupby("category").sum()
# rajouter la transformation dataframe to dict 
# rajouter pie echarts
# rajouter la suite

res = [ {"value" : v, "name":n } for n,v in zip(df_grp["category"].values, df_grp["count"].values) ]
option = {
  "title" : {"text" : 'Referer of a Website', "subtext" : 'Fake Data', "left" : 'center'},
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

st_echarts(options=options, height="600px",)
