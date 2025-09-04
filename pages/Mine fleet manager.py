import pandas as pd
import numpy as np
import streamlit as st

"# :material/conveyor_belt: Mine fleet manager"
"""
Here you can specify machine number, availability and capacity.
"""

class MineOps :
  def __init__(self, dict_opt, mine_supervisors, mine_fleet, mine_task) :
    self.dict_opt = dict_opt
    self.mine_supervisors = mine_supervisors
    self.mine_fleet = mine_fleet
    self.mine_task = mine_task

  def check_dict_opt(self, to_check) :
    if len(self.dict_opt) == 0 :
      st.error("Class MineOps dict_opt is empty, please create one in the List manager page.")
      st.stop()
    for tc,k in zip(["mine_supervisor", "mine_fleet", "mine_task"],["Supervisors", "Machines", "Task"]) :
      if to_check==tc :
        if k in self.dict_opt :
          if len(dict_opt[k]) == 0 :
            st.error(f"No '{k}' type have been created in the List manager page.")
            st.stop()
        else :
          st.error(f"The key {k} does not exist, please create it in the List manager page.")
      
  def config_fleet(self) :
    check_dict_opt("mine_fleet")

