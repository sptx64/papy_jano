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
    self.mtype        = mtype
    self.id           = id
    self.name         = name
    self.capacity     = capacity
    self.comment      = comment
    self.availability = availability

  def machine_to_array(self) :
    return [ self.mtype, self.id, self.name, self.capacity, self.comment, self.availability ], ["Machine type", "Machine ID", "Machine name", "Capacity", "Comment", "Availability"]

  def array_to_machine(self, arr) :
    self.mtype        = arr[0]
    self.id           = arr[1]
    self.name         = arr[2]
    self.capacity     = arr[3]
    self.comment      = arr[4]
    self.availability = arr[5]



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
    st.cache_data.clear()
    time.sleep(3)
    st.rerun()
  
  def create_fleet(self) :
    """
    This function create mine fleet in the MineOps class
    """
    self.check_dict_opt("mine_fleet")
    machine_type = self.dict_opt["Machines"]
    fleet = {}
    for i,mt in enumerate(machine_type) :
      st.write(f"### {mt}")
        
      c                     = st.columns(2)
      nb_mt                 = c[0].number_input(f"Number of :blue-badge[{mt}]", 0, 1000, 2)
      capacity_range        = (0,999999,None)
      default_capacity      = c[1].number_input(f"Default capacity for :blue-badge[{mt}] machines", capacity_range[0], capacity_range[1], capacity_range[2], help="Just to simplify inputting")
      dict_machine_entities = {}
        
      for mt_id in range(nb_mt) :
        c = st.columns(4, vertical_alignment="bottom")
        machine_name         = c[0].text_input(f":blue-badge[{mt}]-{mt_id} name", f"{mt}-{mt_id}")
        machine_capacity     = c[1].number_input(f":blue-badge[{mt}] - {mt_id} capacity (tons)", capacity_range[0], capacity_range[1], default_capacity, help="Can be left empty")
        machine_comment      = c[2].text_input(f":blue-badge[{mt}] - {mt_id} comment", None, placeholder="Any comment you want to link to this machine", help="Can be left empty if no comment")
        machine_availability = c[3].toggle(f":blue-badge[{mt}] - {mt_id} is available", value=True)
        machine_entity       = MachineEntity(mtype=mt, id=mt_id, name=machine_name,
                                         capacity=machine_capacity, comment=machine_comment,
                                         availability=machine_availability)
          
        dict_machine_entities[mt_id] = machine_entity
      st.write("")
      st.write("---")
      st.write("")
      fleet[mt] = dict_machine_entities
    
    if st.button("Save your MineFleet in your MineOps", type="primary") :
      self.mine_fleet = fleet
      self.save_pkl()

  def modify_fleet(self) :
    """
    modify an existing mine_fleet
    """
    if len(list(self.mine_fleet)) == 0 :
      st.warning("It looks like there are no mine fleet already created in that MineOps, start by creating one from scratch!")
      st.stop()
    
    dict_fleet = self.mine_fleet
    res = []
    for mtype in dict_fleet :
      for i in range(len(dict_fleet[mtype])) :
        machine = dict_fleet[mtype][i]
        arr,col_name = machine.machine_to_array()
        res.append(arr)

    df = pd.DataFrame(res, columns=col_name)
    # add a st.fragment
    # add a row deleting availability
    # get the new values in the mine fleet class
    # add a save function
    @st.fragment
    def mine_fleet_editor(df) :
      column_config={
        "Machine type": st.column_config.SelectboxColumn(
            "Machine type",
            help="Type of machine",
            options=[ x for x in self.dict_opt["Machines"] ],
            required=True,
        ),
        "Machine ID" : "Machine ID",
        "Machine name" : "Machine name",
        "Capacity":"Capacity",
        "Comment":"Comment",
        "Availability": "Availability"        
      }
      st.data_editor(df, height=650, hide_index=True, disabled=["ID"], num_rows="dynamic", column_config=column_config )
    mine_fleet_editor(df)
      

  
  def clean_class_dict_opt(self) :
    #cleaning tasks
    present_in_minops_to_keep = [ k for k in self.mine_task if k in self.dict_opt["Task"] ]
    for k in self.mine_task :
      if k not in present_in_minops_to_keep :
        del self.mine_task[k]

    #cleaning supervisors
    present_in_minops_to_keep = [ k for k in self.mine_supervisors if k in self.dict_opt["Supervisors"] ]
    for k in self.mine_supervisors :
      if k not in present_in_minops_to_keep :
        del self.mine_supervisors[k]

    #clean mine fleet
    present_in_minops_to_keep = [ k for k in self.mine_fleet if k in self.dict_opt["Machines"] ]
    for k in self.mine_fleet :
      if k not in present_in_minops_to_keep :
        del self.mine_fleet[k]

  


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
    c = st.columns(3)
    list_task        = c[0].multiselect(":material/assignment_add: Tasks that will be associated to your MineOps", [], [], placeholder="e.g. Basin dragging", accept_new_options=True)
    list_supervisors = c[1].multiselect(":material/man: Supervisors that will be associated to your MineOps", [], [], placeholder="e.g. Simon or Team A", accept_new_options=True)
    list_machines    = c[2].multiselect(":material/conveyor_belt: Machines that will be associated to your MineOps", ["Excavator","Truck","Bull","Drilling machine",], [], placeholder="e.g. Excavator", accept_new_options=True)
    
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
    c = st.columns(3)
    list_task        = c[0].multiselect(":material/assignment_add: Tasks that will be associated to your MineOps", self.dict_opt["Task"], self.dict_opt["Task"], accept_new_options=True)
    list_supervisors = c[1].multiselect(":material/man: Supervisors that will be associated to your MineOps", self.dict_opt["Supervisors"], self.dict_opt["Supervisors"], accept_new_options=True)
    list_machines    = c[2].multiselect(":material/conveyor_belt: Machines that will be associated to your MineOps", self.dict_opt["Machines"], self.dict_opt["Machines"], accept_new_options=True)

    c = st.columns(6)
    if c[1].button("Delete", width="content") :
      file_path=os.path.join(st.session_state.project, f"MineOps - {self.name}.pkl")
      os.remove(file_path)
      st.success("MineOps have been deleted. The app will rerun in 3 seconds.")
      time.sleep(3)
      st.rerun()
    
    if c[0].button("Save", type="primary", width="content") :
      
      dict_opt={
        "Task"        : sorted(list(np.unique(list_task)), key=str.lower),
        "Supervisors" : sorted(list(np.unique(list_supervisors)), key=str.lower),
        "Machines"    : sorted(list(np.unique(list_machines)), key=str.lower),
      }
      
      self.dict_opt = dict_opt
      #cleaning from mine_fleet, mine_task, mine_supervisors classes all the keys that are ne more in dict_opt
      self.clean_class_dict_opt()
      #saving the pkl
      self.save_pkl()


    



      


class Task:
  def __init__(self, id, name, category, responsible, fleet, comments, dependencies, dependency_type, start_date, end_date, progress, projected_end_date, projection_speed, is_critical,) :
    self.id = id
    self.name = name
    self.category = category
    self.responsible = responsible
    self.fleet = fleet
    self.comments = comments
    self.dependencies = dependencies
    self.dependency_type = dependency_type
    self.start_date = start_date
    self.end_date = end_date
    self.progress = progress
    self.projected_end_date = projected_end_date
    self.projection_speed = projection_speed
    self.is_critical = is_critical
    

  
    
    
    
    # """
    # 👋 **Core Task Data Class**
    # - Uses `@dataclass` for auto `__init__`, `__repr__`, etc.
    # - Supports **PERT analysis**, **critical path**, and **progress tracking**.
    # - Added: **Projection Speed** and **P10-P90 Percentiles** for extra risk insights.
    # - Fields grouped into **4 categories** (see below).
    # """
    
    # # 🎯 **GENERAL INFORMATION** (Basics & Identity)
    # id: int  # 🆔 Unique identifier (calculated or entry)
    # name: str  # 📝 Task description (entry)
    # category: str = "Task"  # 🏷️ Category for grouping (entry)
    # responsible: Optional[str] = None  # 👤 Person/team responsible (entry)
    # equipment: Optional[str] = None  # 🛠️ Required equipment (entry)
    # comments: Optional[str] = None  # 💬 Additional notes (entry)
    # dependencies: str = ""  # 🔗 Predecessor task IDs (e.g., "1,2"; entry)
    # dependency_type: str = "FS"  # 🔄 Dependency type (FS, SS, FF, SF; entry)  # ADDED
    # lag: int = 0  # ⏳ Delay/lag (days; entry)
    
    # # 📅 **SCHEDULING** (Dates & Progress)
    # start_date: Optional[date] = None  # 🗓️ Planned start (entry or calculated)
    # end_date: Optional[date] = None  # 🎯 Planned end (entry or calculated)
    # progress: int = 0  # 📊 Completion % (0-100; entry)
    # projected_end_date: Optional[date] = None  # 🔮 Forecasted end from progress (calculated)
    # projection_speed: Optional[float] = None  # 🚀 Speed (% per day; calculated)
    # is_critical: Optional[bool] = None  # ⚡ On critical path? (calculated)
    
    # # ⏱️ **DURATIONS** (PERT Time Estimates)
    # duration_optimistic: Optional[int] = None  # 😊 Best-case days (entry or calculated)
    # duration_pessimistic: Optional[int] = None  # 😰 Worst-case days (entry or calculated)
    # duration_probable: Optional[int] = None  # 🤔 Most likely days (entry or calculated)
    # duration_stochastic: Optional[float] = None  # 📈 PERT mean duration (calculated)
    # duration_days: Optional[int] = None  # 📅 Rounded final days (calculated or entry)
    
    # # ⚠️ **RISK & UNCERTAINTY** (Buffers & Percentiles)
    # standard_deviation: Optional[float] = None  # 📊 Variance measure (calculated)
    # buffer: Optional[float] = None  # 🛡️ Risk buffer (days; calculated)
    # # 🧮 Percentiles (calculated via normal dist from mean & std dev)
    # p10: Optional[float] = None  # 10th percentile (days)
    # p20: Optional[float] = None  # 20th percentile (days)
    # p30: Optional[float] = None  # 30th percentile (days)
    # p40: Optional[float] = None  # 40th percentile (days)
    # p50: Optional[float] = None  # 50th percentile (median; days)
    # p60: Optional[float] = None  # 60th percentile (days)
    # p70: Optional[float] = None  # 70th percentile (days)
    # p80: Optional[float] = None  # 80th percentile (days)
    # p90: Optional[float] = None  # 90th percentile (days)

 
   #Starting date : Calculate the scheduled end date based on start date and duration.
 
 #    def scheduled_end(self) -> Optional[date]:
 #        """
 #        Calculate the scheduled end date based on start date and duration.
        
 #        Priority order:
 #        1. If both start_date and duration_days exist, calculate end date
 #        2. Otherwise, return the manually set end_date
        
 #        Returns:
 #            Optional[date]: Calculated or stored end date
 #        """
 #        if self.start_date and self.duration_days:
 #            return self.start_date + timedelta(days=self.duration_days - 1)
 #        return self.end_date
     
     
 #    #Durée stochastique calculation
 
 #    def get_expected_duration(self) -> Optional[float]:
 #        """
 #        Get the expected duration for the task, fusing calculation logic to cover all cases.
        
 #        Priority order (covers all scenarios from both original methods):
 #        1. If stochastic (PERT calculated) is set: Return it directly.
 #        2. If not set but optimistic/pessimistic are available: Calculate stochastic on the fly (using calculate_stochastic_duration logic), set the field, and return it.
 #        3. If calculation not possible but duration_days is set: Return duration_days as float.
 #        4. Otherwise: Return None.
        
 #        This fused version ensures the output is identical to running both methods separately, but in one step.
 #        Formulas identical to originals: PERT = (O + 4M + P)/6 where M = probable or (O+P)/2.
        
 #        Returns:
 #            Optional[float]: Expected duration in days.
 #        """
 #        if self.duration_stochastic is not None:
 #            # Priority 1: Use existing stochastic
 #            return self.duration_stochastic
        
 #        # Priority 2: Calculate stochastic if possible (fusing calculate_stochastic_duration logic)
 #        if self.duration_optimistic is not None and self.duration_pessimistic is not None:
 #            # Use provided probable or calculate average (same as calculate_stochastic_duration)
 #            probable = self.duration_probable or ((self.duration_optimistic + self.duration_pessimistic) / 2)
 #            # Cache probable if not set
 #            if self.duration_probable is None:
 #                self.duration_probable = probable
 #            # Calculate and set stochastic
 #            self.duration_stochastic = (self.duration_optimistic + 4 * probable + self.duration_pessimistic) / 6
 #            return self.duration_stochastic
        
 #        # Priority 3: Fallback to duration_days if set
 #        elif self.duration_days is not None:
 #            return float(self.duration_days)
        
 #        # No valid data: None
 #        return None

 
 #    #Calcul Task Duration with start date and end date
 
 #    def get_actual_duration(self) -> Optional[int]:
 #        """
 #        Calculate actual duration if task is completed.
        
 #        Returns:
 #            Optional[int]: Actual duration in days if both dates exist
 #        """
 #        if self.start_date and self.end_date:
 #            return (self.end_date - self.start_date).days + 1
 #        return None



 #     #- Calcul End date projection and Projection speed
     
 #    def get_remaining_duration(self) -> Optional[float]:
 #        """
 #        Calculate remaining duration based on current progress, using formulas for Projection Speed and Projected End Date.
        
 #        Formulas (based on your model, translated to English):
 #        - Work Done Days = duration_stochastic * (progress / 100)
 #        - Elapsed Days = (date.today() - start_date).days if start_date else 0
 #        - Projection Speed = Work Done Days / Elapsed Days (if applicable)
 #        - Projected Total Duration = duration_stochastic / Projection Speed
 #        - Projected End Date = start_date + Projected Total Duration (updates projected_end_date)
 #        - Remaining Duration = max(0, Projected Total Duration - Elapsed Days)
        
 #        This uses Projection Speed for dynamic estimates.
 #        Falls back to expected * (100 - progress)/100 if calculations not applicable.
        
 #        Returns:
 #            Optional[float]: Remaining duration in days.
 #        """
 #        expected = self.get_expected_duration()
 #        if not expected or not self.start_date:
 #            # Not applicable, fallback to simple remaining estimate
 #            return expected * (100 - self.progress) / 100 if expected and self.progress < 100 else 0.0
        
 #        # Calculate Work Done Days
 #        work_done_days = expected * (self.progress / 100.0)
        
 #        # Calculate Elapsed Days (using int for simplicity)
 #        days_elapsed = (date.today() - self.start_date).days
        
 #        if days_elapsed <= 0:
 #            # Not started, full expected remaining
 #            return expected * (100 - self.progress) / 100 if self.progress < 100 else 0.0
        
 #        # Calculate Projection Speed
 #        if work_done_days > 0:
 #            projection_speed = work_done_days / days_elapsed
 #        else:
 #            # No progress, assume neutral speed
 #            projection_speed = 1.0
        
 #        # Calculate Projected Total Duration
 #        if projection_speed > 0:
 #            projected_total_duration = expected / projection_speed
 #        else:
 #            projected_total_duration = expected
        
 #        # Update projected_end_date and projection_speed
 #        self.projected_end_date = self.start_date + timedelta(days=int(round(projected_total_duration)))
 #        self.projection_speed = round(projection_speed, 2)
        
 #        # Calculate Remaining Duration
 #        remaining_duration = max(0.0, projected_total_duration - days_elapsed)
        
 #        # If completed, remaining is 0
 #        if self.progress >= 100:
 #            return 0.0
        
 #        return round(remaining_duration, 2)


     
 #    #Calculate End-date.
 
 #    def get_stochastic_end_date(self) -> Optional[date]:
 #        """
 #        Calculate end date based on stochastic duration.
        
 #        Returns:
 #            Optional[date]: Calculated end date using stochastic duration
 #        """
 #        if self.start_date and self.duration_stochastic:
 #            return self.start_date + timedelta(days=int(round(self.duration_stochastic)))
 #        elif self.start_date and self.duration_days:
 #            return self.start_date + timedelta(days=self.duration_days - 1)
 #        return self.end_date

 
 # #? 
 
 #    def get_standard_deviation(self) -> Optional[float]:
 #        """
 #        Get the standard deviation for the task.
        
 #        Returns:
 #            Optional[float]: Standard deviation value
 #        """
 #        return self.standard_deviation

 

 
 #    def to_dict(self):
 #        """
 #        Convert Task object to dictionary for JSON serialization.
        
 #        Returns:
 #            dict: Task data as dictionary with proper date formatting
 #        """
 #        d = asdict(self)
 #        # Convert date objects to ISO format strings for JSON compatibility
 #        d["start_date"] = self.start_date.isoformat() if self.start_date else None
 #        d["end_date"] = self.end_date.isoformat() if self.end_date else None
 #        d["projected_end_date"] = self.projected_end_date.isoformat() if self.projected_end_date else None
 #        return d


 
 #    #DEPENDENCY
 
 #    def get_dependency_ids(self) -> List[int]:
 #        """
 #        Parse dependency string and return list of task IDs.
        
 #        Supports both comma and semicolon separators.
        
 #        Returns:
 #            List[int]: List of predecessor task IDs
 #        """
 #        if not self.dependencies:
 #            return []
 #        # Normalize separators and parse
 #        deps_str = self.dependencies.replace(";", ",")
 #        return [int(x.strip()) for x in deps_str.split(",") if x.strip().isdigit()]


     
 #    #Check data
 
 #    def is_complete_for_calculation(self) -> bool:
 #        """
 #        Check if task has sufficient information for scheduling calculations.
        
 #        Returns:
 #            bool: True if task can be used in calculations
 #        """
 #        has_dates = self.start_date and self.end_date
 #        has_start_duration = self.start_date and self.duration_days
 #        has_end_duration = self.end_date and self.duration_days
 #        return has_dates or has_start_duration or has_end_duration
          
        
        
        
          
          
          
          
      
    



