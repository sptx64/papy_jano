import pandas as pd
import numpy as np
import streamlit as st

"# :material/conveyor_belt: Mine fleet manager"
"""
Here you can specify machine number, availability and capacity.
"""

class MachineEntity :
  def __init__(self, mtype, id, name, capacity, comment, availability) :
    self.mtype = machine_type
    self.id = machine_id
    self.name = machine_name
    self.capacity = machine_capacity
    self.comment = machine_comment
    self.availability = machine_availability


class MineOps :
  def __init__(self, dict_opt, mine_supervisors={}, mine_fleet={}, mine_task={}) :
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
          if len(self.dict_opt[k]) == 0 :
            st.error(f"No '{k}' type have been created in the List manager page.")
            st.stop()
        else :
          st.error(f"The key {k} does not exist, please create it in the List manager page.")
      
  def fleet_setup(self) :
    check_dict_opt("mine_fleet")
    machine_type = self.dict_opt["Machines"]
    fleet={}
    t = st.tabs(machine_types)
    for i,mt in enumerate(machine_type) :
      with t[i] :
        f"### {mt}"
        f"Setup the number of {mt}, availability, "
        c = st.columns(3)
        nb_mt = c[0].number_input(f"Number of {mt}", 0, 1000, 2)
        capacity_range = (0,999999,None)
        default_capacity = c[1].number_input(f"Default capacity for {mt} machines", capacity[0], capacity[1], capacity[2], help="Just to simplify inputting")
        dict_machine_entities = {}
        for mt_id in nb_mt :
          c = st.columns(4)
          machine_name = c[0].text_input(f"{mt} - id:{mt_id} name", f"{mt}-{mt_id}")
          machine_capacity = c[1].number_input(f"{mt} - id:{mt_id} capacity (tons)", capacity[0], capacity[1], default_capacity, help="Can be left empty")
          machine_comment = c[2].text_input(f"{mt} - id:{mt_id} comment", None, placeholder="Any comment you want to link to this machine", help="Can be left empty if no comment")
          machine_availability = c[3].checkbox(f"{mt} - id:{mt_id} is available", value=True)
          machine_entity = MachineEntity(mtype=mt, id=mt_id, name=machine_name,
                                         capacity=machine_capacity, comment=machine_comment,
                                         available=machine_availability)
          dict_machine_entities[mt_id] = machine_entity
        fleet[mt] = dict_machine_entities
    
    if st.button("Name your Mine Fleet", type="primary") :
      @st.dialog("Mine Fleet naming", width="medium")
      def mine_fleet_save(fleet) :
        "### Saving your mine fleet"
        c = st.columns(2)
        if c[1].toggle("Overwrite an existing fleet?") :
          input_opt = (list(self.mine_fleet), False, "Select the MineFleet you want to overwrite") if self.mine_fleet is not None and len(self.mine_fleet)>0 else ([], True, "There are no MineFleet existing in your project")
          mfleet_name = c[0].selectbox("MineFleet to overwrite", input_opt[0], disabled=input_opt[1], help=input_opt[2])
        else :
          mfleet_name = c[0].text_input("Create a new MineFleet", placeholder="Your new mine fleet name")
        
        if mfleet_name is not None and mfleet_name != "" :
          if st.button("Save mine fleet") :
            self.mine_fleet[mfleet_name] = fleet
            st.session_state.mine_ops = self
            
            # save mine fleet
        else :
          st.info("Please select/enter a valid name")
            
            
            
            
          
        
        
        
          
          
          
          
      
    

