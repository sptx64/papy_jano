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


def calculate_buffer(predecessor_std=None, duration_optimistic, duration_pessimistic, buffer=None, is_critical, coefficient_critical, multiplier_multi_dependencies) :
  # buffer de la tâche
  if duration_optimistic is None or duration_pessimistic is None:
    return buffer  # Keep existing value if no input data (as before)
    
  if standard_deviation is None:
    calculate_standard_deviation(task)  # Assumes this sets task.standard_deviation
            
  if standard_deviation is None:
    return buffer
        
  num_predecessors = len( get_dependency_ids() )
  is_multi_dependency = num_predecessors > 1
        
  if is_critical:
    if predecessor_std is not None:
      combined_std = np.sqrt(predecessor_std**2 + standard_deviation**2)
    else:
      combined_std = standard_deviation
      buffer = coefficient_critical * combined_std
  else:
    buffer = coefficient_non_critical * standard_deviation
        
  if is_multi_dependency and buffer is not None:
    buffer *= multiplier_multi_dependencies
            
    return buffer
        

def calculate_projected_end_date(start_date, progress, duration_stochastic, projected_end_date):
  if not start_date or not duration_stochastic or progress == 0:
    return projected_end_date  # Keep existing value
        
  work_done_days = duration_stochastic * (progress / 100)      
  days_elapsed   = (date.today() - start_date).days
  
  if days_elapsed <= 0:
    days_elapsed = 1  # Minimum 1 day to avoid division by zero
        
  execution_speed = work_done_days / days_elapsed
  if execution_speed > 0:
    real_duration = duration_stochastic / execution_speed
  else:
    real_duration = duration_stochastic
    
  projected_end_date = start_date + timedelta(days=int(round(real_duration)))
  return projected_end_date


def calculate_beta_standard_deviation(duration_optimistic, duration_probable, duration_pessimistic, lambda_value):
  if (duration_optimistic is None or duration_pessimistic is None or
    duration_probable is None or duration_optimistic >= duration_pessimistic):
    return None        
  O=duration_optimistic;  P=duration_pessimistic; M=duration_probable      
  
  alpha = 1 + lambda_value * (M - O) / (P - O)
  beta_param = 1 + lambda_value * (P - M) / (P - O)
        
  var_Y = (alpha * beta_param) / ((alpha + beta_param) ** 2 * (alpha + beta_param + 1))
  var_X = (P - O) ** 2 * var_Y
        
  standard_deviation = np.sqrt(var_X)
  return standard_deviation


def calculate_beta_percentiles(lambda_value):
  if (task.duration_optimistic is None or task.duration_pessimistic is None or
    task.duration_probable is None or task.duration_optimistic >= task.duration_pessimistic):
    return

  O, P, M = task.duration_optimistic, task.duration_pessimistic, task.duration_probable
        
  alpha = 1 + lambda_value * (M - O) / (P - O)
  beta_param = 1 + lambda_value * (P - M) / (P - O)
        
  # Step 2: Define quantiles
  quantiles = {
    0.10: 'p10', 0.20: 'p20', 0.30: 'p30', 0.40: 'p40',
    0.50: 'p50', 0.60: 'p60', 0.70: 'p70', 0.80: 'p80', 0.90: 'p90'
  }
        
  # Step 3: Calculate and assign to task
  for q, attr in quantiles.items():
    y_q = beta.ppf(q, alpha, beta_param)
    setattr(task, attr, round(O + (P - O) * y_q, 2))  # Fix: Assigner à task


def calculate_standard_deviation(standard_deviation, duration_optimistic, duration_pessimistic):
  if duration_optimistic is None or duration_pessimistic is None:
    return standard_deviation
  std_dev = (duration_pessimistic - duration_optimistic) / 6 # Pourquoi 6 ???
  return std_dev

def _calculate_start_from_dependencies():
  dep_ids = get_dependency_ids()
  if not dep_ids:
    return start_date

  latest_end = None
  dep_type = dependency_type
  lag_days = task.lag

  for dep_id in dep_ids:
    parent = self.tasks.get(dep_id) # améliorer si task dépendante est terminée
    if not parent:
      continue  # Skip non-existent predecessor

    parent_date = None

    if dep_type == "FS":  # Finish to Start (most common)
      parent_date = parent.scheduled_end()
      if parent_date:
        parent_date += timedelta(days=1 + lag_days)
                    
    elif dep_type == "SS":  # Start to Start
      parent_date = parent.start_date
      if parent_date:
        parent_date += timedelta(days=lag_days)
                    
    elif dep_type == "FF":  # Finish to Finish
      parent_date = parent.scheduled_end()
      if parent_date and task.duration_days:
        parent_date = parent_date - timedelta(days=task.duration_days - 1) + timedelta(days=lag_days)
                    
    elif dep_type == "SF":  # Start to Finish (rare)
      parent_date = parent.start_date
      if parent_date and task.duration_days:
        parent_date = parent_date - timedelta(days=task.duration_days - 1) + timedelta(days=lag_days)

    if parent_date and (latest_end is None or parent_date > latest_end):
      latest_end = parent_date

    return latest_end


def find_critical_path(self) -> List[int]:
  if not self.tasks:
    return []

  # === FORWARD PASS - Calculate Earliest Times ===
  earliest_start = {}  # Earliest possible start date for each task
  earliest_finish = {}  # Earliest possible finish date for each task
        
  # Process tasks in dependency order (topological sort)
  sorted_tasks = self._topological_sort()
        
  for task_id in sorted_tasks:
    task = self.tasks[task_id]
            
  # Calculate earliest start date
  dep_ids = task.get_dependency_ids()
  if not dep_ids:
    # No dependencies - can start immediately
    earliest_start[task_id] = task.start_date or date.today()
  else:
    # Must wait for all predecessors to finish
    max_pred_finish = None
    for dep_id in dep_ids:
      if dep_id in earliest_finish:
        pred_finish = earliest_finish[dep_id]
        if max_pred_finish is None or pred_finish > max_pred_finish:
          max_pred_finish = pred_finish    
          
      if max_pred_finish:
        earliest_start[task_id] = max_pred_finish + timedelta(days=1)
      else:
        earliest_start[task_id] = task.start_date or date.today()
            
  # Calculate earliest finish date
  duration = task.duration_days or (int(task.duration_stochastic) if task.duration_stochastic else 1)
  earliest_finish[task_id] = earliest_start[task_id] + timedelta(days=duration - 1)

  # === BACKWARD PASS - Calculate Latest Times ===
  latest_start = {}   # Latest allowable start date
  latest_finish = {}  # Latest allowable finish date
        
  # Find project end date (latest finish of all tasks)
  if earliest_finish:
    project_end = max(earliest_finish.values())
  else:
    return []
        
  # Process tasks in reverse dependency order
  for task_id in reversed(sorted_tasks):
    task = self.tasks[task_id]
            
    # Find all successor tasks
    successors = [t.id for t in self.tasks.values() if task_id in t.get_dependency_ids()]
            
    if not successors:
      # No successors - this is a final task
      latest_finish[task_id] = project_end
    else:
      # Must finish before earliest successor starts
      min_succ_start = None
      for succ_id in successors:
        if succ_id in latest_start:
          succ_start = latest_start[succ_id]
          if min_succ_start is None or succ_start < min_succ_start:
            min_succ_start = succ_start
                
      if min_succ_start:
        latest_finish[task_id] = min_succ_start - timedelta(days=1)
      else:
        latest_finish[task_id] = project_end
            
    # Calculate latest start date
    duration = task.duration_days or (int(task.duration_stochastic) if task.duration_stochastic else 1)
    latest_start[task_id] = latest_finish[task_id] - timedelta(days=duration - 1)

  # === IDENTIFY CRITICAL TASKS ===
  # Critical tasks have zero slack (earliest = latest)
  critical_tasks = []
  for task_id in self.tasks.keys():
    if (task_id in earliest_start and task_id in latest_start and
      earliest_start[task_id] == latest_start[task_id]):
      critical_tasks.append(task_id)
        
  return critical_tasks


def identify_critical_tasks(self, critical_path: List[int]):
  for task in self.tasks.values():
    task.is_critical = task.id in critical_path

    
# def auto_calculate_all_tasks(self, max_iterations: int = 10):
        
#   # ... PHASE 1: PERT CALCULATIONS (mise à jour)
#   for task in self.tasks.values():
#     if task.duration_optimistic is not None and task.duration_pessimistic is not None:
#       task.get_expected_duration()  # Lump calculate_stochastic_duration here
#       if task.standard_deviation is None:
#         self.calculate_beta_standard_deviation(task, lambda_value=4.0)  # Add this call
#         self.calculate_beta_percentiles(task, lambda_value=4.0)  # Add this call
#         if task.duration_days is None and task.duration_stochastic is not None:
#           task.duration_days = int(round(task.duration_stochastic))


#   changes_made = True
#   iteration = 0

#   while changes_made and iteration < max_iterations:
#     changes_made = False
#     iteration += 1

#     sorted_tasks = self._topological_sort()

#     for task_id in sorted_tasks:
#       task = self.tasks[task_id]
                
#       old_start = task.start_date
#       old_end = task.end_date
#       old_duration = task.duration_days

#       if task.get_dependency_ids() and not task.start_date:
#         calculated_start = self._calculate_start_from_dependencies(task)
#         if calculated_start and calculated_start != task.start_date:
#           task.start_date = calculated_start
#           changes_made = True
                
          
#       if task.start_date and task.duration_days and not task.end_date:
#         new_end = task.start_date + timedelta(days=task.duration_days - 1)
#         if new_end != task.end_date:
#           task.end_date = new_end
#           changes_made = True

#       elif task.end_date and task.duration_days and not task.start_date:
#         new_start = task.end_date - timedelta(days=task.duration_days - 1)
#         if new_start != task.start_date:
#           task.start_date = new_start
#           changes_made = True

#       # Calculate duration from start to end
#       elif task.start_date and task.end_date and not task.duration_days:
#         duration = (task.end_date - task.start_date).days + 1
#         if duration != task.duration_days:
#           task.duration_days = duration
#           changes_made = True

#         # === PHASE 3: CRITICAL PATH ANALYSIS ===
#         # Identify which tasks are critical for project completion
#         try:
#             critical_path = self.find_critical_path()
#             self.identify_critical_tasks(critical_path)
#         except Exception as e:
#             # Handle critical path calculation errors gracefully
#             if hasattr(st, 'warning'):
#                 st.warning(f"Erreur lors du calcul du chemin critique: {e}")

#         # === PHASE 4: BUFFER CALCULATIONS ===
#         # Calculate risk buffers based on task criticality and dependencies
#         sorted_tasks = self._topological_sort()
#         for task_id in sorted_tasks:
#             task = self.tasks[task_id]
            
#             # Only calculate buffer if we have PERT data and buffer not set
#             if (task.duration_optimistic is not None and 
#                 task.duration_pessimistic is not None and 
#                 task.buffer is None):
                
#                 # Get predecessor standard deviation for critical tasks
#                 predecessor_std = None
#                 if task.is_critical and task.get_dependency_ids():
#                     dep_id = task.get_dependency_ids()[0]
#                     dep_task = self.tasks.get(dep_id)
#                     if dep_task and dep_task.standard_deviation is not None:
#                         predecessor_std = dep_task.standard_deviation
                
#                 self.calculate_buffer(task, predecessor_std)

#         # === PHASE 5: PROGRESS PROJECTIONS ===
#         for task in self.tasks.values():
#             # Always recalculate projections and remaining
#             if task.progress > 0 and task.start_date and task.duration_stochastic:
#                 self.calculate_projected_end_date(task)
#                 task.get_remaining_duration()  # Add this call for Projection Speed integration

#         # === PHASE 6: SCHEDULING UPDATES ONLY (NO BUFFER INTEGRATION) ===
#         # Simply update end_date for consistency, without adding buffer
#         for task in self.tasks.values():
#             # Only update end_date for scheduling consistency - do NOT add buffer to duration
#             if task.start_date and task.duration_days:
#                 # Update end_date based on duration_days (without buffer integration)
#                 task.end_date = task.start_date + timedelta(days=task.duration_days - 1)
            
#             # Buffer remains a separate risk indicator, not integrated into actual schedule
