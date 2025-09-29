import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import os
import pickle

def test():
  ""


def get_save_folder() :
  return "files"


@st.cache_data
def read_minops(fpath) :
  with open(fpath, "rb") as f :
    mo = pickle.load(f)
  return mo

def load_opt_pkl(start_with, location) :
  """
  Function to load a pkl file that contain all list options
  The function return the dictionary if it's found locally on the "location" folder by looking for a file starting with "start_with"
  If a file is found then it's stored in the st.session_state
  If multiple files are found it will flag a warning for the user and the first file will be loaded
  if no file are found it will store an empty dictionary in st.session_state
  ---
  warning on the streamlit public cloud, the files stay only 24h, then are deleted.
  it will have a better behavior when run locally
  """
  
  if "dict_opt" not in st.session_state :
    #checking if the folder in input exists
    if not os.path.exists(location) :
      st.error(f":material/error: There was an error : the folder {location} does not exist")
      st.stop()
  
    #storing a list of all files that start with the key start_with
    list_files = [ os.path.join(location,x) for x in os.listdir(location) if x.startswith(start_with) and x.endswith(".pkl")]
    if len(list_files) >= 1 :
      # if there are files that starts with start_with and ends with pkl
      with open(list_files[0], "rb") as f :
        st.session_state.dict_opt = pickle.load(f)
      
      if len(list_files) == 1 :
        st.toast(f"The file {list_files[0]} have been imported successfully and cached", icon=":material/check_small:")
      else :
        st.toast(f":orange-badge[Warning!] there was more than one file in the folder {location} with a name starting with {start_with}. The first file encoutered {list_file[0]} has been loaded by default.", icon=":material/warning:")
        
    else :
      st.session_state.dict_opt = {}


def scheduled_end(start_date, duration_days):
  if start_date and self.duration_days:
    return start_date + timedelta(days=duration_days - 1)
  else :
    return self.end_date


def get_expected_duration(duration_stochastic=None, duration_optimistic=None, duration_probable=None, duration_pessimistic=None, duration_days=None):
  if duration_stochastic is not None:
    return duration_stochastic
    
  if duration_optimistic is not None and duration_pessimistic is not None:
    probable            = duration_probable or ((duration_optimistic + duration_pessimistic) / 2)
    duration_probable   = probable if duration_probable is None else duration_probable
    duration_stochastic = (duration_optimistic + 4 * probable + duration_pessimistic) / 6
    return duration_stochastic
        
  elif duration_days is not None:
    return float(duration_days)

  return None


def get_actual_duration(start_date, end_date) :
  if start_date is not None and end_date is not None:
    return (end_date - start_date).days + 1
  else :
    return None


def get_remaining_duration(duration_stochastic, duration_optimistic, duration_probable, duration_pessimistic, duration_days, progress, start_date):
  expected = get_expected_duration(duration_stochastic, duration_optimistic,
                                   duration_probable, duration_pessimistic,
                                   duration_days)
  
  if not expected or not start_date:
    return expected * (100 - progress) / 100 if expected and progress < 100 else 0.0
        
  work_done_days = expected * (progress / 100.0)
  days_elapsed   = (date.today() - start_date).days
        
  if days_elapsed <= 0:
    return expected * (100 - progress) / 100 if progress < 100 else 0.0
        
  projection_speed         = work_done_days / days_elapsed if work_done_days > 0 else projection_speed = 1.0
  projected_total_duration = expected / projection_speed if projection_speed > 0 else expected
        
  projected_end_date = start_date + timedelta(days=int(round(projected_total_duration)))
  projection_speed   = round(projection_speed, 2)  
  remaining_duration = max(0.0, projected_total_duration - days_elapsed)    
  
  if progress >= 100:
    return 0.0    
  return round(remaining_duration, 2)


def get_stochastic_end_date(start_date, end_date, duration_stochastic, duration_days):
  if start_date and duration_stochastic:
    return start_date + timedelta(days=int(round(duration_stochastic)))
  elif start_date and duration_days:
    return start_date + timedelta(days=duration_days - 1)
  return end_date


def get_standard_deviation(standard_deviation):
  return standard_deviation


def to_dict(start_date, end_date, projected_end_date):
  d = asdict()
  d["start_date"]         = start_date.isoformat() if start_date else None
  d["end_date"]           = end_date.isoformat() if end_date else None
  d["projected_end_date"] = projected_end_date.isoformat() if projected_end_date else None
  return d

def get_dependency_ids(dependencies):
  if not dependencies:
    return []
  else :
    deps_str = dependencies.replace(";", ",")
    return [int(x.strip()) for x in deps_str.split(",") if x.strip().isdigit()]


def is_complete_for_calculation(start_date, end_date, duration_days):
  has_dates          = start_date and end_date
  has_start_duration = start_date and duration_days
  has_end_duration   = end_date   and duration_days
  return has_dates or has_start_duration or has_end_duration


def next_id(k_task):
  return max(self.tasks.keys(), default=0) + 1 # à revoir!


def calculate_buffer(self, task: Task, predecessor_std: Optional[float] = None) -> Optional[float]:
  if task.duration_optimistic is None or task.duration_pessimistic is None:
            return task.buffer  # Keep existing value if no input data (as before)
    
  if task.standard_deviation is None:
            self.calculate_standard_deviation(task)  # Assumes this sets task.standard_deviation
            
  if task.standard_deviation is None:
            return task.buffer
        
  num_predecessors = len(task.get_dependency_ids())
  is_multi_dependency = num_predecessors > 1
        
  if task.is_critical:
    if predecessor_std is not None:
      combined_std = np.sqrt(predecessor_std**2 + task.standard_deviation**2)
    else:
      combined_std = task.standard_deviation
      task.buffer = self.coefficient_critical * combined_std
  else:
    task.buffer = self.coefficient_non_critical * task.standard_deviation
        
  if is_multi_dependency and task.buffer is not None:
    task.buffer *= self.multiplier_multi_dependencies
            
    return task.buffer
        
