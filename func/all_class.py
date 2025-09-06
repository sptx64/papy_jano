import pandas as pd
import numpy as np
import streamlit as st
import os
from .data import get_save_folder
import time
import pickle

"""
this file store all classes available
"""

class MachineEntity :
  def __init__(self, name, mtype, id, capacity, comment, availability) :
    """
    This function initiate the MachineEntity class called in MineOps
    """
    # self.name         = name
    self.mtype        = machine_type
    self.id           = machine_id
    self.name         = name
    self.capacity     = machine_capacity
    self.comment      = machine_comment
    self.availability = machine_availability


class MineOps :
  def __init__(self, name=None, dict_opt={}, mine_supervisors={}, mine_fleet={}, mine_task={}) :
    """
    This function initiate the MineOps class
    """
    self.name             = name
    self.dict_opt         = dict_opt
    self.mine_supervisors = mine_supervisors
    self.mine_fleet       = mine_fleet
    self.mine_task        = mine_task

  def check_dict_opt(self, to_check) :
    """
    This function checks if dict_opt is empty then if mine_supervisors, mine_fleet or mine_task sub classes are empty 
    """
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

  def save_pkl(self) :
    #saving in the cache of streamlit (st.session_state)
    st.session_state.MineOps = self
    
    fpath_name = os.path.join(st.session_state.project, f"MineOps - {self.name}.pkl")
    with open(fpath_name, "wb") as f :
      pickle.dump(self, f)
      st.toast("The MineOps have been saved!", icon=":material/check_small:")
      st.success("Saved! The app will rerun in 3 seconds.")
      time.sleep(3)
      st.rerun()
  
  def create_fleet(self) :
    """
    This function create mine fleet in the MineOps class
    """
    check_dict_opt("mine_fleet")
    machine_type = self.dict_opt["Machines"]
    fleet = {}
    t = st.tabs(machine_types)
    for i,mt in enumerate(machine_type) :
      with t[i] :
        f"### {mt}"
        f"Setup the number of {mt}, availability, "
        
        c                     = st.columns(3)
        nb_mt                 = c[0].number_input(f"Number of {mt}", 0, 1000, 2)
        capacity_range        = (0,999999,None)
        default_capacity      = c[1].number_input(f"Default capacity for {mt} machines", capacity[0], capacity[1], capacity[2], help="Just to simplify inputting")
        dict_machine_entities = {}
        
        for mt_id in nb_mt :
          c = st.columns(4)
          machine_name         = c[0].text_input(f"{mt} - id:{mt_id} name", f"{mt}-{mt_id}")
          machine_capacity     = c[1].number_input(f"{mt} - id:{mt_id} capacity (tons)", capacity[0], capacity[1], default_capacity, help="Can be left empty")
          machine_comment      = c[2].text_input(f"{mt} - id:{mt_id} comment", None, placeholder="Any comment you want to link to this machine", help="Can be left empty if no comment")
          machine_availability = c[3].checkbox(f"{mt} - id:{mt_id} is available", value=True)
          machine_entity       = MachineEntity(mtype=mt, id=mt_id, name=machine_name,
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
            #get folder save name
            #save pkl to folder
            #with open('person.pkl', 'wb') as file:  # 'wb' for binary write mode
              #pickle.dump(person, file)

        else :
          st.info("Please select/enter a valid name")

  def create_dict_opt(self) :
    mops_name = st.text_input("New MineOps name", None, placeholder="YOUR NEW MINEOPS NAME")
    if mops_name is None or mops_name == "" :
      st.info("Enter a name for your new MineOps")
      st.stop()
    mops_prefix = "MineOps - "
    list_existing_mops = [ x.replace(mops_prefix,"").replace(".pkl","") for x in os.listdir(st.session_state.project) if x.startswith(mops_prefix) and x.endswith(".pkl") ]
    if mops_name in list_existing_mops :
      st.warning("This MineOps name already exists.")
      st.stop()
    
    list_task        = st.multiselect(":material/assignment_add: Task options that will be available in other modules", [], [], accept_new_options=True)
    list_supervisors = st.multiselect(":material/man: Supervisors options that will be available in other modules", [], [], accept_new_options=True)
    list_machines    = st.multiselect(":material/conveyor_belt: Type of machines that will be available in other modules", ["Excavator","Truck","Bull","Drilling machine",], [], accept_new_options=True)
    
    #one unique save button that will save all lists in a dictionary and deleting any duplicate option
    if st.button("Save", type="primary", help="Save your new MineOps") :
      dict_opt={
        "Task"        : sorted(list(np.unique(list_task)), key=str.lower),
        "Supervisors" : sorted(list(np.unique(list_supervisors)), key=str.lower),
        "Machines"    : sorted(list(np.unique(list_machines)), key=str.lower),
      }
      self.dict_opt = dict_opt
      self.name     = mops_name
      self.save_pkl()

  def modify_dict_opt(self) :
    list_task        = st.multiselect(":material/assignment_add: Task options that will be available in other modules", self.dict_opt["Task"], self.dict_opt["Task"], accept_new_options=True)
    list_supervisors = st.multiselect(":material/man: Supervisors options that will be available in other modules", self.dict_opt["Supervisors"], self.dict_opt["Supervisors"], accept_new_options=True)
    list_machines    = st.multiselect(":material/conveyor_belt: Type of machines that will be available in other modules", self.dict_opt["Machines"], self.dict_opt["Machines"], accept_new_options=True)

    c = st.columns(6)
    if c[1].button("Delete", use_container_width=True) :
      file_path=os.path.join(st.session_state.project, f"MineOps - {self.name}.pkl")
      os.remove(file_path)
      st.success("MineOps have been deleted. The app will rerun in 3 seconds.")
      time.sleep(3)
      st.rerun()
    
    if c[0].button("Save", type="primary", use_container_width=True) :
      dict_opt={
        "Task"        : sorted(list(np.unique(list_task)), key=str.lower),
        "Supervisors" : sorted(list(np.unique(list_supervisors)), key=str.lower),
        "Machines"    : sorted(list(np.unique(list_machines)), key=str.lower),
      }
      
      self.dict_opt = dict_opt
      self.save_pkl()
    
    
      
      # #storing it locally for next sessions
      # full_name = os.path.join(folder, f"{fname_begin}.pkl")
      # with open(full_name, "wb") as f :
      #   pickle.dump(dict_opt, f)
      #   st.toast("The options have been saved!", icon=":material/check_small:")
      #   st.success("Saved! The app will rerun in 3 seconds.")
      #   time.sleep(3)
      #   st.rerun()


      
            
            
          
        
        
        
          
          
          
          
      
    



