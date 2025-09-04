"""
Advanced Task Management System for Mine Planning
=================================================

This application provides a comprehensive task management system specifically designed for mine planning projects.
It uses PERT (Program Evaluation and Review Technique) methodology and critical path analysis to optimize
project schedules and manage risks through buffer calculations.

Key Features:
- PERT-based stochastic duration calculations
- Critical path identification
- Risk-based buffer calculations
- Task dependency management
- Progress tracking and projection
- Interactive Streamlit interface

Author: Mine Planning Team
Version: 1.0
"""
from scipy.stats import beta
import json
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Optional, List
import pandas as pd
import streamlit as st
import numpy as np
from itertools import product
import random
from scipy.stats import beta  # Si pas déjà là
import numpy as np  # Si pas déjà là
import pandas as pd  # Si pas déjà là
import matplotlib.pyplot as plt
from io import BytesIO


# ============================================================================================
# CONFIGURATION CONSTANTS
# ============================================================================================

# File paths for data persistence
SAVE_PATH = Path("data")  # Directory for saving project data
SAVE_FILE = SAVE_PATH / "mine_plan.json"  # Main project file

# ============================================================================================
# DATA MODEL - TASK DEFINITION
# ============================================================================================


    ### 1-1 - DATA TABLEAU
# ============================================================================================
@dataclass
class Task:
    """
    👋 **Core Task Data Class**
    - Uses `@dataclass` for auto `__init__`, `__repr__`, etc.
    - Supports **PERT analysis**, **critical path**, and **progress tracking**.
    - Added: **Projection Speed** and **P10-P90 Percentiles** for extra risk insights.
    - Fields grouped into **4 categories** (see below).
    """
    
    # 🎯 **GENERAL INFORMATION** (Basics & Identity)
    id: int  # 🆔 Unique identifier (calculated or entry)
    name: str  # 📝 Task description (entry)
    category: str = "Task"  # 🏷️ Category for grouping (entry)
    responsible: Optional[str] = None  # 👤 Person/team responsible (entry)
    equipment: Optional[str] = None  # 🛠️ Required equipment (entry)
    comments: Optional[str] = None  # 💬 Additional notes (entry)
    dependencies: str = ""  # 🔗 Predecessor task IDs (e.g., "1,2"; entry)
    dependency_type: str = "FS"  # 🔄 Dependency type (FS, SS, FF, SF; entry)  # ADDED
    lag: int = 0  # ⏳ Delay/lag (days; entry)
    
    # 📅 **SCHEDULING** (Dates & Progress)
    start_date: Optional[date] = None  # 🗓️ Planned start (entry or calculated)
    end_date: Optional[date] = None  # 🎯 Planned end (entry or calculated)
    progress: int = 0  # 📊 Completion % (0-100; entry)
    projected_end_date: Optional[date] = None  # 🔮 Forecasted end from progress (calculated)
    projection_speed: Optional[float] = None  # 🚀 Speed (% per day; calculated)
    is_critical: Optional[bool] = None  # ⚡ On critical path? (calculated)
    
    # ⏱️ **DURATIONS** (PERT Time Estimates)
    duration_optimistic: Optional[int] = None  # 😊 Best-case days (entry or calculated)
    duration_pessimistic: Optional[int] = None  # 😰 Worst-case days (entry or calculated)
    duration_probable: Optional[int] = None  # 🤔 Most likely days (entry or calculated)
    duration_stochastic: Optional[float] = None  # 📈 PERT mean duration (calculated)
    duration_days: Optional[int] = None  # 📅 Rounded final days (calculated or entry)
    
    # ⚠️ **RISK & UNCERTAINTY** (Buffers & Percentiles)
    standard_deviation: Optional[float] = None  # 📊 Variance measure (calculated)
    buffer: Optional[float] = None  # 🛡️ Risk buffer (days; calculated)
    # 🧮 Percentiles (calculated via normal dist from mean & std dev)
    p10: Optional[float] = None  # 10th percentile (days)
    p20: Optional[float] = None  # 20th percentile (days)
    p30: Optional[float] = None  # 30th percentile (days)
    p40: Optional[float] = None  # 40th percentile (days)
    p50: Optional[float] = None  # 50th percentile (median; days)
    p60: Optional[float] = None  # 60th percentile (days)
    p70: Optional[float] = None  # 70th percentile (days)
    p80: Optional[float] = None  # 80th percentile (days)
    p90: Optional[float] = None  # 90th percentile (days)


 ###1-3 Calcul du Tableau
# ============================================================================================

 
   #Starting date : Calculate the scheduled end date based on start date and duration.
 
    def scheduled_end(self) -> Optional[date]:
        """
        Calculate the scheduled end date based on start date and duration.
        
        Priority order:
        1. If both start_date and duration_days exist, calculate end date
        2. Otherwise, return the manually set end_date
        
        Returns:
            Optional[date]: Calculated or stored end date
        """
        if self.start_date and self.duration_days:
            return self.start_date + timedelta(days=self.duration_days - 1)
        return self.end_date
     
     
    #Durée stochastique calculation
 
    def get_expected_duration(self) -> Optional[float]:
        """
        Get the expected duration for the task, fusing calculation logic to cover all cases.
        
        Priority order (covers all scenarios from both original methods):
        1. If stochastic (PERT calculated) is set: Return it directly.
        2. If not set but optimistic/pessimistic are available: Calculate stochastic on the fly (using calculate_stochastic_duration logic), set the field, and return it.
        3. If calculation not possible but duration_days is set: Return duration_days as float.
        4. Otherwise: Return None.
        
        This fused version ensures the output is identical to running both methods separately, but in one step.
        Formulas identical to originals: PERT = (O + 4M + P)/6 where M = probable or (O+P)/2.
        
        Returns:
            Optional[float]: Expected duration in days.
        """
        if self.duration_stochastic is not None:
            # Priority 1: Use existing stochastic
            return self.duration_stochastic
        
        # Priority 2: Calculate stochastic if possible (fusing calculate_stochastic_duration logic)
        if self.duration_optimistic is not None and self.duration_pessimistic is not None:
            # Use provided probable or calculate average (same as calculate_stochastic_duration)
            probable = self.duration_probable or ((self.duration_optimistic + self.duration_pessimistic) / 2)
            # Cache probable if not set
            if self.duration_probable is None:
                self.duration_probable = probable
            # Calculate and set stochastic
            self.duration_stochastic = (self.duration_optimistic + 4 * probable + self.duration_pessimistic) / 6
            return self.duration_stochastic
        
        # Priority 3: Fallback to duration_days if set
        elif self.duration_days is not None:
            return float(self.duration_days)
        
        # No valid data: None
        return None

 
    #Calcul Task Duration with start date and end date
 
    def get_actual_duration(self) -> Optional[int]:
        """
        Calculate actual duration if task is completed.
        
        Returns:
            Optional[int]: Actual duration in days if both dates exist
        """
        if self.start_date and self.end_date:
            return (self.end_date - self.start_date).days + 1
        return None



     #- Calcul End date projection and Projection speed
     
    def get_remaining_duration(self) -> Optional[float]:
        """
        Calculate remaining duration based on current progress, using formulas for Projection Speed and Projected End Date.
        
        Formulas (based on your model, translated to English):
        - Work Done Days = duration_stochastic * (progress / 100)
        - Elapsed Days = (date.today() - start_date).days if start_date else 0
        - Projection Speed = Work Done Days / Elapsed Days (if applicable)
        - Projected Total Duration = duration_stochastic / Projection Speed
        - Projected End Date = start_date + Projected Total Duration (updates projected_end_date)
        - Remaining Duration = max(0, Projected Total Duration - Elapsed Days)
        
        This uses Projection Speed for dynamic estimates.
        Falls back to expected * (100 - progress)/100 if calculations not applicable.
        
        Returns:
            Optional[float]: Remaining duration in days.
        """
        expected = self.get_expected_duration()
        if not expected or not self.start_date:
            # Not applicable, fallback to simple remaining estimate
            return expected * (100 - self.progress) / 100 if expected and self.progress < 100 else 0.0
        
        # Calculate Work Done Days
        work_done_days = expected * (self.progress / 100.0)
        
        # Calculate Elapsed Days (using int for simplicity)
        days_elapsed = (date.today() - self.start_date).days
        
        if days_elapsed <= 0:
            # Not started, full expected remaining
            return expected * (100 - self.progress) / 100 if self.progress < 100 else 0.0
        
        # Calculate Projection Speed
        if work_done_days > 0:
            projection_speed = work_done_days / days_elapsed
        else:
            # No progress, assume neutral speed
            projection_speed = 1.0
        
        # Calculate Projected Total Duration
        if projection_speed > 0:
            projected_total_duration = expected / projection_speed
        else:
            projected_total_duration = expected
        
        # Update projected_end_date and projection_speed
        self.projected_end_date = self.start_date + timedelta(days=int(round(projected_total_duration)))
        self.projection_speed = round(projection_speed, 2)
        
        # Calculate Remaining Duration
        remaining_duration = max(0.0, projected_total_duration - days_elapsed)
        
        # If completed, remaining is 0
        if self.progress >= 100:
            return 0.0
        
        return round(remaining_duration, 2)


     
    #Calculate End-date.
 
    def get_stochastic_end_date(self) -> Optional[date]:
        """
        Calculate end date based on stochastic duration.
        
        Returns:
            Optional[date]: Calculated end date using stochastic duration
        """
        if self.start_date and self.duration_stochastic:
            return self.start_date + timedelta(days=int(round(self.duration_stochastic)))
        elif self.start_date and self.duration_days:
            return self.start_date + timedelta(days=self.duration_days - 1)
        return self.end_date

 
 #? 
 
    def get_standard_deviation(self) -> Optional[float]:
        """
        Get the standard deviation for the task.
        
        Returns:
            Optional[float]: Standard deviation value
        """
        return self.standard_deviation

 

 
    def to_dict(self):
        """
        Convert Task object to dictionary for JSON serialization.
        
        Returns:
            dict: Task data as dictionary with proper date formatting
        """
        d = asdict(self)
        # Convert date objects to ISO format strings for JSON compatibility
        d["start_date"] = self.start_date.isoformat() if self.start_date else None
        d["end_date"] = self.end_date.isoformat() if self.end_date else None
        d["projected_end_date"] = self.projected_end_date.isoformat() if self.projected_end_date else None
        return d


 
    #DEPENDENCY
 
    def get_dependency_ids(self) -> List[int]:
        """
        Parse dependency string and return list of task IDs.
        
        Supports both comma and semicolon separators.
        
        Returns:
            List[int]: List of predecessor task IDs
        """
        if not self.dependencies:
            return []
        # Normalize separators and parse
        deps_str = self.dependencies.replace(";", ",")
        return [int(x.strip()) for x in deps_str.split(",") if x.strip().isdigit()]


     
    #Check data
 
    def is_complete_for_calculation(self) -> bool:
        """
        Check if task has sufficient information for scheduling calculations.
        
        Returns:
            bool: True if task can be used in calculations
        """
        has_dates = self.start_date and self.end_date
        has_start_duration = self.start_date and self.duration_days
        has_end_duration = self.end_date and self.duration_days
        return has_dates or has_start_duration or has_end_duration

# ============================================================================================
# TASK MANAGER - CORE BUSINESS LOGIC
# ============================================================================================

class TaskManager:
    """
    Main business logic class for managing mine planning tasks.
    
    This class handles:
    - Task CRUD operations
    - PERT calculations
    - Critical path analysis
    - Buffer calculations
    - Data persistence
    - Dependency management
    """
    
    def __init__(self):
        """
        Initialize TaskManager with default settings and load existing data.
        """
        # === BUFFER CALCULATION COEFFICIENTS ===
        # These coefficients are used in risk management calculations
        self.coefficient_non_critical = 0.5  # Buffer coefficient for non-critical tasks
        self.coefficient_critical = 1.3      # Buffer coefficient for critical tasks
        self.multiplier_multi_dependencies = 1.2  # Additional multiplier for tasks with multiple dependencies
        
        # === DATA STORAGE ===
        self.tasks: Dict[int, Task] = {}  # Main task storage dictionary
        
        # === INITIALIZATION ===
        SAVE_PATH.mkdir(exist_ok=True)  # Ensure data directory exists
        
        # Load existing data or create sample data
        if not self.load():
            self.create_sample()

    # ========================================================================================
    # DATA PERSISTENCE METHODS
    # ========================================================================================

    def next_id(self) -> int:
        """
        Generate next available task ID.
        
        Returns:
            int: Next unique task ID
        """
        return max(self.tasks.keys(), default=0) + 1


 # SAVE TABLE
    def save(self):
        """
        Save all tasks to JSON file.
        
        Serializes the entire task database to a JSON file for persistence.
        """
        # Convert all tasks to dictionary format
        raw = {"tasks": {tid: t.to_dict() for tid, t in self.tasks.items()}}
        
        # Write to file with proper encoding
        SAVE_FILE.write_text(
            json.dumps(raw, indent=2, ensure_ascii=False), 
            encoding="utf-8"
        )

    def load(self) -> bool:  # CORRECTED
        """
        Load tasks from JSON file.
        
        Returns:
            bool: True if loading was successful, False otherwise
        """
        if not SAVE_FILE.exists():
            return False
        
        try:
            # Read and parse JSON data
            raw = json.loads(SAVE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return False

        # Clear existing tasks
        self.tasks = {}
        
        # Reconstruct Task objects from saved data
        for k, d in raw.get("tasks", {}).items():
            # Handle legacy dependency format (list vs string)
            dependencies = d.get("dependencies", "")
            if isinstance(dependencies, list):
                dependencies = ",".join(map(str, dependencies))

            # Create Task object with all stored data (including new field dependency_type)
            self.tasks[int(k)] = Task(
                id=int(k),
                name=d["name"],
                category=d.get("category", "Task"),
                responsible=d.get("responsible"),
                equipment=d.get("equipment"),
                comments=d.get("comments"),
                dependencies=dependencies,
                dependency_type=d.get("dependency_type", "FS"),  # Added
                lag=d.get("lag", 0),  # Added
                # Scheduling
                start_date=date.fromisoformat(d.get("start_date")) if d.get("start_date") else None,
                end_date=date.fromisoformat(d.get("end_date")) if d.get("end_date") else None,
                progress=d.get("progress", 0),
                projected_end_date=date.fromisoformat(d.get("projected_end_date")) if d.get("projected_end_date") else None,
                projection_speed=d.get("projection_speed"),
                is_critical=d.get("is_critical"),
                # Durations
                duration_optimistic=d.get("duration_optimistic"),
                duration_pessimistic=d.get("duration_pessimistic"),
                duration_probable=d.get("duration_probable"),
                duration_stochastic=d.get("duration_stochastic"),
                duration_days=d.get("duration_days"),
                # Risk & Uncertainty
                standard_deviation=d.get("standard_deviation"),
                buffer=d.get("buffer"),
                p10=d.get("p10"), p20=d.get("p20"), p30=d.get("p30"), p40=d.get("p40"),  # Added percentiles
                p50=d.get("p50"), p60=d.get("p60"), p70=d.get("p70"), p80=d.get("p80"), p90=d.get("p90"),
            )
        return True

     # DATA EXEMPLE 
    def create_sample(self):  # CORRECTED - Now uses named arguments like the dataclass expects
        """
        Create sample mine planning tasks for demonstration.
        
        This creates a realistic mine development project with typical tasks:
        - Site preparation
        - Access road construction  
        - Initial drilling
        - Equipment installation
        """
        td = date.today()
        self.tasks = {
            1: Task(  # Now uses named args to match dataclass
                id=1, name="Préparation du site", start_date=td, duration_optimistic=5, duration_pessimistic=10, duration_probable=7, progress=30, responsible="Équipe A", equipment="Bulldozer"
            ),
            2: Task(
                id=2, name="Construction route d'accès", duration_optimistic=7, duration_pessimistic=12, duration_probable=9, progress=15, dependencies="1", responsible="Équipe B", equipment="Excavatrice"
            ),
            3: Task(
                id=3, name="Forage initial", duration_optimistic=10, duration_pessimistic=18, duration_probable=14, progress=50, dependencies="1", responsible="Équipe C", equipment="Foreuse"
            ),
            4: Task(
                id=4, name="Installation équipements", duration_optimistic=3, duration_pessimistic=8, duration_probable=5, progress=0, dependencies="2,3", responsible="Équipe D", equipment="Grue"
            ),
        }

    # ========================================================================================
    # BUFFER CALCULATION METHODS
    # ========================================================================================


    def calculate_buffer(self, task: Task, predecessor_std: Optional[float] = None) -> Optional[float]:
        """
        Calculate risk buffer for task based on criticality and dependencies, matching your exact description.
        
        Description (translated and matched):
        ## **1-4 Calculating Buffer:**
        
        - **For non-critical task:**
          Buffer = Non-critical Coefficient × Task's Standard Deviation
        
        - **For critical task:**
          Combined Standard Deviation = √(Predecessor Standard Deviation² + Current Task Standard Deviation²)
          Buffer = Critical Coefficient × Combined Standard Deviation
        
        - **For multi-dependency task:**
          Buffer = Buffer calculated above × Multi-dependencies Multiplier
        
        Buffer calculation rules:
        - Critical tasks: Use higher coefficient (self.coefficient_critical) and combine with predecessor variance
        - Non-critical tasks: Use lower coefficient (self.coefficient_non_critical)
        - Multi-dependency tasks: Apply additional multiplier (self.multiplier_multi_dependencies) to the calculated buffer
        
        Args:
            task (Task): Task to calculate buffer for
            predecessor_std (Optional[float]): Standard deviation from predecessor tasks (for critical tasks)
            
        Returns:
            Optional[float]: Calculated buffer in days (matches description output)
        """
        if task.duration_optimistic is None or task.duration_pessimistic is None:
            return task.buffer  # Keep existing value if no input data (as before)
            
        # Ensure standard deviation is calculated (required for buffer formulas)
        if task.standard_deviation is None:
            self.calculate_standard_deviation(task)  # Assumes this sets task.standard_deviation
            
        if task.standard_deviation is None:
            return task.buffer
        
        # Analyze task characteristics for multi-dependencies
        num_predecessors = len(task.get_dependency_ids())
        is_multi_dependency = num_predecessors > 1
        
        if task.is_critical:
            # For critical task: Combined Std = √(Pred Std² + Task Std²), Buffer = Critical Coeff × Combined Std
            if predecessor_std is not None:
                combined_std = np.sqrt(predecessor_std**2 + task.standard_deviation**2)
            else:
                combined_std = task.standard_deviation
            task.buffer = self.coefficient_critical * combined_std
        else:
            # For non-critical task: Buffer = Non-Critical Coeff × Task Std
            task.buffer = self.coefficient_non_critical * task.standard_deviation
        
        # For multi-dependency task: Buffer = Buffer calculated above × Multi-Dependencies Multiplier
        if is_multi_dependency and task.buffer is not None:
            task.buffer *= self.multiplier_multi_dependencies
            
        return task.buffer


    def calculate_projected_end_date(self, task: Task) -> Optional[date]:
        """
        Calculate projected end date based on current progress and execution speed.
        
        This method analyzes actual vs planned progress to project realistic completion dates.
        
        Args:
            task (Task): Task to calculate projection for
            
        Returns:
            Optional[date]: Projected completion date
        """
        # Only calculate if we have necessary data and some progress
        if not task.start_date or not task.duration_stochastic or task.progress == 0:
            return task.projected_end_date  # Keep existing value
        
        # Calculate work completed in equivalent days
        work_done_days = task.duration_stochastic * (task.progress / 100)
        
        # Calculate actual elapsed time
        days_elapsed = (date.today() - task.start_date).days
        
        # Handle edge cases for newly started tasks
        if days_elapsed <= 0:
            days_elapsed = 1  # Minimum 1 day to avoid division by zero
        
        # Calculate execution speed (work days per calendar day)
        execution_speed = work_done_days / days_elapsed
        
        # Project total duration based on current execution speed
        if execution_speed > 0:
            real_duration = task.duration_stochastic / execution_speed
        else:
            # Fallback to original estimate if no measurable progress
            real_duration = task.duration_stochastic
        
        # Calculate projected end date
        task.projected_end_date = task.start_date + timedelta(days=int(round(real_duration)))
        
        return task.projected_end_date
     

                                              #### Ecart type 

     

    def calculate_beta_standard_deviation(self, task: Task, lambda_value: float = 4.0) -> Optional[float]:
        """
        Calculate standard deviation using Beta distribution (PERT methodology).
        """
        if (task.duration_optimistic is None or task.duration_pessimistic is None or
            task.duration_probable is None or task.duration_optimistic >= task.duration_pessimistic):
            return None
        
        O, P, M = task.duration_optimistic, task.duration_pessimistic, task.duration_probable
        
        # Step 1: Calculate α and β (Beta distribution parameters)
        alpha = 1 + lambda_value * (M - O) / (P - O)
        beta_param = 1 + lambda_value * (P - M) / (P - O)
        
        # Step 2: Calculate variance of scaled Beta RV
        var_Y = (alpha * beta_param) / ((alpha + beta_param) ** 2 * (alpha + beta_param + 1))
        
        # Step 3: Scale to actual duration variance
        var_X = (P - O) ** 2 * var_Y
        
        # Step 4: Standard deviation
        standard_deviation = np.sqrt(var_X)
        task.standard_deviation = standard_deviation  # Fix: Assigner à task
        return standard_deviation
     
                                  ##### Calculs Percentiles 
    def calculate_beta_percentiles(self, task: Task, lambda_value: float = 4.0):
        """
        Calculate P10 to P90 percentiles using Beta distribution.
        """
        if (task.duration_optimistic is None or task.duration_pessimistic is None or
            task.duration_probable is None or task.duration_optimistic >= task.duration_pessimistic):
            return
        
        O, P, M = task.duration_optimistic, task.duration_pessimistic, task.duration_probable
        
        # Step 1: Calculate α and β (same as above)
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


         

    def calculate_standard_deviation(self, task: Task) -> Optional[float]:
        """
        Simple fallback for standard deviation (if Beta not used).
        Formula: (Pessimistic - Optimistic) / 6
        """
        if task.duration_optimistic is None or task.duration_pessimistic is None:
            return task.standard_deviation
        
        # Simple PERT std dev
        std_dev = (task.duration_pessimistic - task.duration_optimistic) / 6
        task.standard_deviation = std_dev
        return std_dev
         
    # ========================================================================================
    # DEPENDENCY AND SCHEDULING METHODS
    # ========================================================================================

    def _calculate_start_from_dependencies(self, task: Task) -> Optional[date]:
        """
        Calculate task start date based on predecessor dependencies.
        
        Supports different dependency types:
        - FS (Finish to Start): Task starts after predecessor finishes
        - SS (Start to Start): Task starts when predecessor starts
        - FF (Finish to Finish): Task finishes when predecessor finishes
        - SF (Start to Finish): Task finishes when predecessor starts
        
        Args:
            task (Task): Task to calculate start date for
            
        Returns:
            Optional[date]: Calculated start date
        """
        dep_ids = task.get_dependency_ids()
        if not dep_ids:
            return task.start_date  # No dependencies, keep current value

        latest_end = None
        dep_type = task.dependency_type.upper()
        lag_days = task.lag

        # Process each predecessor
        for dep_id in dep_ids:
            parent = self.tasks.get(dep_id)
            if not parent:
                continue  # Skip non-existent predecessors

            parent_date = None

            # Calculate date based on dependency type
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

            # Keep the latest constraint date
            if parent_date and (latest_end is None or parent_date > latest_end):
                latest_end = parent_date

        return latest_end

    # ========================================================================================
    # CRITICAL PATH ANALYSIS
    # ========================================================================================

    def find_critical_path(self) -> List[int]:
        """
        Implement Critical Path Method (CPM) to find the longest path through the project.
        
        The critical path determines the minimum project duration and identifies tasks
        that cannot be delayed without affecting the project completion date.
        
        Algorithm:
        1. Forward pass: Calculate earliest start/finish times
        2. Backward pass: Calculate latest start/finish times  
        3. Identify critical tasks: Tasks where early = late (zero slack)
        
        Returns:
            List[int]: List of task IDs on the critical path
        """
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
        """
        Mark tasks as critical or non-critical based on critical path analysis.
        
        Args:
            critical_path (List[int]): List of task IDs on the critical path
        """
        for task in self.tasks.values():
            task.is_critical = task.id in critical_path

    # ========================================================================================
    # COMPREHENSIVE CALCULATION ENGINE
    # ========================================================================================

    def auto_calculate_all_tasks(self, max_iterations: int = 10):
        """
        Comprehensive calculation engine that automatically computes all task metrics.
        
        This is the main calculation method that orchestrates all computations:
        1. PERT duration calculations
        2. Date and dependency resolution
        3. Critical path analysis
        4. Buffer calculations
        5. Progress projections
        
        Uses iterative approach to resolve complex dependency chains.
        
        Args:
            max_iterations (int): Maximum iterations to prevent infinite loops
        """
        
        # ... PHASE 1: PERT CALCULATIONS (mise à jour)
        for task in self.tasks.values():
            if task.duration_optimistic is not None and task.duration_pessimistic is not None:
                # Use fused method for stochastic duration
                task.get_expected_duration()  # Lump calculate_stochastic_duration here
                # Calculate standard deviation using Beta (more accurate)
                if task.standard_deviation is None:
                    self.calculate_beta_standard_deviation(task, lambda_value=4.0)  # Add this call
                # Calculate percentiles using Beta
                self.calculate_beta_percentiles(task, lambda_value=4.0)  # Add this call
                # Rounded duration
                if task.duration_days is None and task.duration_stochastic is not None:
                    task.duration_days = int(round(task.duration_stochastic))

        # === PHASE 2: DATE AND DEPENDENCY RESOLUTION ===
        # Iteratively resolve dates and dependencies until convergence
        changes_made = True
        iteration = 0

        while changes_made and iteration < max_iterations:
            changes_made = False
            iteration += 1

            # Process tasks in dependency order to minimize iterations
            sorted_tasks = self._topological_sort()

            for task_id in sorted_tasks:
                task = self.tasks[task_id]
                
                # Store previous values to detect changes
                old_start = task.start_date
                old_end = task.end_date
                old_duration = task.duration_days

                # === START DATE CALCULATION ===
                # Calculate start date from dependencies if not manually set
                if task.get_dependency_ids() and not task.start_date:
                    calculated_start = self._calculate_start_from_dependencies(task)
                    if calculated_start and calculated_start != task.start_date:
                        task.start_date = calculated_start
                        changes_made = True

                # === DATE-DURATION CONSISTENCY ===
                # Ensure start date, end date, and duration are consistent
                
                # Calculate end date from start + duration
                if task.start_date and task.duration_days and not task.end_date:
                    new_end = task.start_date + timedelta(days=task.duration_days - 1)
                    if new_end != task.end_date:
                        task.end_date = new_end
                        changes_made = True

                # Calculate start date from end - duration
                elif task.end_date and task.duration_days and not task.start_date:
                    new_start = task.end_date - timedelta(days=task.duration_days - 1)
                    if new_start != task.start_date:
                        task.start_date = new_start
                        changes_made = True

                # Calculate duration from start to end
                elif task.start_date and task.end_date and not task.duration_days:
                    duration = (task.end_date - task.start_date).days + 1
                    if duration != task.duration_days:
                        task.duration_days = duration
                        changes_made = True

        # === PHASE 3: CRITICAL PATH ANALYSIS ===
        # Identify which tasks are critical for project completion
        try:
            critical_path = self.find_critical_path()
            self.identify_critical_tasks(critical_path)
        except Exception as e:
            # Handle critical path calculation errors gracefully
            if hasattr(st, 'warning'):
                st.warning(f"Erreur lors du calcul du chemin critique: {e}")

        # === PHASE 4: BUFFER CALCULATIONS ===
        # Calculate risk buffers based on task criticality and dependencies
        sorted_tasks = self._topological_sort()
        for task_id in sorted_tasks:
            task = self.tasks[task_id]
            
            # Only calculate buffer if we have PERT data and buffer not set
            if (task.duration_optimistic is not None and 
                task.duration_pessimistic is not None and 
                task.buffer is None):
                
                # Get predecessor standard deviation for critical tasks
                predecessor_std = None
                if task.is_critical and task.get_dependency_ids():
                    dep_id = task.get_dependency_ids()[0]
                    dep_task = self.tasks.get(dep_id)
                    if dep_task and dep_task.standard_deviation is not None:
                        predecessor_std = dep_task.standard_deviation
                
                self.calculate_buffer(task, predecessor_std)

        # === PHASE 5: PROGRESS PROJECTIONS ===
        for task in self.tasks.values():
            # Always recalculate projections and remaining
            if task.progress > 0 and task.start_date and task.duration_stochastic:
                self.calculate_projected_end_date(task)
                task.get_remaining_duration()  # Add this call for Projection Speed integration

        # === PHASE 6: SCHEDULING UPDATES ONLY (NO BUFFER INTEGRATION) ===
        # Simply update end_date for consistency, without adding buffer
        for task in self.tasks.values():
            # Only update end_date for scheduling consistency - do NOT add buffer to duration
            if task.start_date and task.duration_days:
                # Update end_date based on duration_days (without buffer integration)
                task.end_date = task.start_date + timedelta(days=task.duration_days - 1)
            
            # Buffer remains a separate risk indicator, not integrated into actual schedule

    # ========================================================================================
    # UTILITY AND HELPER METHODS
    # ========================================================================================

    def _topological_sort(self) -> List[int]:
        """
        Perform topological sort of tasks based on dependencies.
        
        This ensures tasks are processed in the correct order, with predecessor
        tasks always processed before their dependents. Uses depth-first search
        with cycle detection.
        
        Returns:
            List[int]: Task IDs in topological order
        """
        result = []
        visited = set()        # Permanently visited nodes
        temp_visited = set()   # Temporarily visited nodes (for cycle detection)

        def visit(task_id: int):
            """Recursive depth-first search visitor."""
            if task_id in temp_visited:
                return  # Cycle detected - ignore to prevent infinite recursion
            if task_id in visited:
                return  # Already processed

            temp_visited.add(task_id)

            # Visit all dependencies first (depth-first)
            task = self.tasks.get(task_id)
            if task:
                for dep_id in task.get_dependency_ids():
                    if dep_id in self.tasks:
                        visit(dep_id)

            # Mark as permanently visited and add to result
            temp_visited.remove(task_id)
            visited.add(task_id)
            result.append(task_id)

        # Process all tasks
        for task_id in self.tasks.keys():
            if task_id not in visited:
                visit(task_id)

        return result

    # ========================================================================================
    # VALIDATION AND ERROR CHECKING
    # ========================================================================================

    def validate_task_data(self) -> List[str]:
        """
        Comprehensive validation of all task data.
        
        Checks for:
        - Required fields (task names)
        - Circular dependencies
        - Invalid dependency references
        - Data consistency issues
        
        Returns:
            List[str]: List of validation error messages
        """
        errors = []
        
        for task in self.tasks.values():
            # Check required fields
            if not task.name.strip():
                errors.append(f"Tâche {task.id}: Le nom est obligatoire")

            # Check for circular dependencies
            if self._has_circular_dependency(task.id):
                errors.append(f"Tâche {task.id}: Dépendance circulaire détectée")

            # Validate dependency references
            for dep_id in task.get_dependency_ids():
                if dep_id not in self.tasks:
                    errors.append(f"Tâche {task.id}: Dépendance {dep_id} inexistante")

        return errors

    def _has_circular_dependency(self, task_id: int, visited: set = None) -> bool:
        """
        Detect circular dependencies using depth-first search.
        
        A circular dependency exists if a task depends (directly or indirectly)
        on itself, creating an impossible scheduling situation.
        
        Args:
            task_id (int): Task ID to check
            visited (set): Set of already visited task IDs
            
        Returns:
            bool: True if circular dependency detected
        """
        if visited is None:
            visited = set()

        if task_id in visited:
            return True  # Found a cycle

        visited.add(task_id)
        task = self.tasks.get(task_id)
        
        if task:
            # Recursively check all dependencies
            for dep_id in task.get_dependency_ids():
                if self._has_circular_dependency(dep_id, visited.copy()):
                    return True

        return False

# ============================================================================================
# DATA INTERCHANGE UTILITIES
# ============================================================================================

def update_tasks_from_editor(tm: TaskManager, edited_df: pd.DataFrame):  # CORRECTED VERSION
    """
    Update TaskManager from Streamlit data editor DataFrame, matching the new English keys and full dataclass.
    
    Handles data type conversions, PRESERVES calculated fields AND user input if not modified, and adds new fields like category and percentiles.
    
    Args:
        tm (TaskManager): TaskManager instance to update
        edited_df (pd.DataFrame): DataFrame from Streamlit data editor with English keys
    """
    
    # === IDENTIFY TASK CHANGES ===
    # Determine which tasks to keep, update, or delete based on 'id'
    new_ids = set()
    for _, row in edited_df.iterrows():
        if pd.notna(row.get("id")):
            new_ids.add(int(row["id"]))

    # Remove deleted tasks
    for tid in list(tm.tasks.keys()):
        if tid not in new_ids:
            del tm.tasks[tid]

    # === PROCESS EACH ROW FROM EDITOR ===
    for _, row in edited_df.iterrows():
        # Skip rows without a name (but only if id exists and is not just empty)
        name_value = str(row.get("name", "")).strip()
        if not name_value and pd.isna(row.get("id", "")):
            continue  # Skip completely empty rows

        tid = int(row["id"]) if pd.notna(row.get("id")) else tm.next_id()
        existing_task = tm.tasks.get(tid)  # Get existing if available

        # === DATE PARSING UTILITIES ===
        def parse_date(date_str, fallback_date=None):
            """Robust date parsing; returns fallback if invalid."""
            if pd.isna(date_str) or not str(date_str).strip():
                return fallback_date  # Use existing or None
            try:
                date_str = str(date_str).strip()
                if len(date_str) == 10 and '-' in date_str:
                    return date.fromisoformat(date_str)
            except (ValueError, TypeError):
                return fallback_date  # Invalid -> fallback
            return None

        # === NUMERIC PARSING UTILITIES ===
        def safe_int(value, fallback_int=None):
            """Safe int; returns fallback if invalid."""
            if pd.isna(value) or value == "":
                return fallback_int
            try:
                return int(float(value))
            except (ValueError, TypeError):
                return fallback_int

        def safe_float(value, fallback_float=None):
            """Safe float; returns fallback if invalid."""
            if pd.isna(value) or value == "":
                return fallback_float
            try:
                return float(value)
            except (ValueError, TypeError):
                return fallback_float

        # === DETERMINE NEW VALUES WITH PRESERVATION ===
        # For each field: If row has valid value, use it; else preserve existing (if any)
        new_name = name_value or (existing_task.name if existing_task else "")
        new_category = str(row.get("category", "")).strip() or (existing_task.category if existing_task else "Task")
        new_responsible = str(row.get("responsible", "")).strip() or (existing_task.responsible if existing_task else None)
        new_equipment = str(row.get("equipment", "")).strip() or (existing_task.equipment if existing_task else None)
        new_comments = str(row.get("comments", "")).strip() or (existing_task.comments if existing_task else None)
        new_dependencies = str(row.get("dependencies", "")).strip() or (existing_task.dependencies if existing_task else "")
        new_dependency_type = str(row.get("dependency_type", "")).strip() or (existing_task.dependency_type if existing_task else "FS")
        new_lag = safe_int(row.get("lag"), existing_task.lag if existing_task else 0) or 0
        new_progress = safe_int(row.get("progress"), existing_task.progress if existing_task else 0) or 0
        new_start_date = parse_date(row.get("start_date"), existing_task.start_date if existing_task else None)
        new_end_date = parse_date(row.get("end_date"), existing_task.end_date if existing_task else None)
        new_duration_optimistic = safe_int(row.get("duration_optimistic"), existing_task.duration_optimistic if existing_task else None)
        new_duration_pessimistic = safe_int(row.get("duration_pessimistic"), existing_task.duration_pessimistic if existing_task else None)
        new_duration_probable = safe_int(row.get("duration_probable"), existing_task.duration_probable if existing_task else None)
        new_duration_days = safe_int(row.get("duration_days"), existing_task.duration_days if existing_task else None)
        
        # Always preserve calculated fields (they will be recalculated by auto_calculate_all_tasks)
        preserved_stochastic = existing_task.duration_stochastic if existing_task else None
        preserved_std = existing_task.standard_deviation if existing_task else None
        preserved_buffer = existing_task.buffer if existing_task else None
        preserved_projected = existing_task.projected_end_date if existing_task else None
        preserved_projection_speed = existing_task.projection_speed if existing_task else None
        preserved_is_critical = existing_task.is_critical if existing_task else None
        preserved_p10 = existing_task.p10 if existing_task else None
        preserved_p20 = existing_task.p20 if existing_task else None
        preserved_p30 = existing_task.p30 if existing_task else None
        preserved_p40 = existing_task.p40 if existing_task else None
        preserved_p50 = existing_task.p50 if existing_task else None
        preserved_p60 = existing_task.p60 if existing_task else None
        preserved_p70 = existing_task.p70 if existing_task else None
        preserved_p80 = existing_task.p80 if existing_task else None
        preserved_p90 = existing_task.p90 if existing_task else None

        # === CREATE/UPDATE TASK ===
        task = Task(
            id=tid,
            name=new_name,
            category=new_category,
            responsible=new_responsible,
            equipment=new_equipment,
            comments=new_comments,
            dependencies=new_dependencies,
            dependency_type=new_dependency_type,
            lag=new_lag,
            start_date=new_start_date,
            end_date=new_end_date,
            progress=new_progress,
            projected_end_date=preserved_projected,
            projection_speed=preserved_projection_speed,
            is_critical=preserved_is_critical,
            duration_optimistic=new_duration_optimistic,
            duration_pessimistic=new_duration_pessimistic,
            duration_probable=new_duration_probable,
            duration_stochastic=preserved_stochastic,
            duration_days=new_duration_days,
            standard_deviation=preserved_std,
            buffer=preserved_buffer,
            p10=preserved_p10, p20=preserved_p20, p30=preserved_p30, p40=preserved_p40,
            p50=preserved_p50, p60=preserved_p60, p70=preserved_p70, p80=preserved_p80, p90=preserved_p90,
        )

        tm.tasks[int(tid)] = task

# ============================================================================================
# STREAMLIT USER INTERFACE
# ============================================================================================

# Configure Streamlit page
st.set_page_config(page_title="⚒️ Gestionnaire de Tâches", layout="wide")

def main():
    """
    Main application entry point.
    
    Sets up the Streamlit interface with navigation and session state management.
    """
    # === NAVIGATION SIDEBAR ===
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.selectbox(
        "Choisir une page",
        ["📊 Gestion des Tâches", "📈 Visualisations & Statistiques", "🧩 Simulation de Projet"]  # <-- Ajouté ici si pas déjà fait
    )

    # === SESSION STATE MANAGEMENT ===
    # Initialize TaskManager in session state for persistence across interactions
    if "task_manager" not in st.session_state:
        st.session_state.task_manager = TaskManager()

    tm = st.session_state.task_manager

    # === PAGE ROUTING ===
    if page == "📊 Gestion des Tâches":
        show_task_management_page(tm)
    elif page == "📈 Visualisations & Statistiques":
        try:
            # Import visualization module (external dependency)
            from visualizations import show_visualizations_page  # Garde ton import original
            show_visualizations_page(tm)
        except Exception as e:
            st.error("❌ Erreur lors du chargement des visualisations :")
            st.code(str(e))
    elif page == "🧩 Simulation de Projet":
        try:
            from simulation import show_simulation_page  # <-- CORRIGÉ : nom de ton fichier sans espaces
            show_simulation_page(tm)  # <-- CORRIGÉ : nom de la fonction sans espaces
        except Exception as e:
            st.error(f"❌ Erreur lors du chargement de la simulation : {e}")

def show_task_management_page(tm: TaskManager):
    """
    Main task management interface.
    
    This function creates the comprehensive task management UI including:
    - Interactive data editor
    - Calculation controls
    - Configuration options
    - Summary statistics
    - Help documentation
    
    Args:
        tm (TaskManager): TaskManager instance
    """

# === CONFIGURATION INITIALE (Sidebar, Headers, etc.) ===
    # ... votre code existant ici ...

    # === CALCULS DES COEFFS ===
    coeff_non   = round(tm.coefficient_non_critical, 2)
    coeff_crit  = round(tm.coefficient_critical, 2)
    coeff_multi = round(tm.multiplier_multi_dependencies, 2)

    # === ACCORDÉON DES FORMULES (au-dessus du tableau) ===
    with st.expander("📐 **Formules & Calculs des Colonnes**", expanded=False):

        st.markdown(f"""
        Cet accordéon décrit **chaque colonne du tableau**, avec son type (manuel ou calculé),  
        sa formule et son rôle dans la planification.  

        *(Valeurs actuelles : Non-Crit = **{coeff_non}**, Crit = **{coeff_crit}**, Multi = **{coeff_multi}**)*  

        ---

        ## 🔎 1. Informations Générales
        | Colonne       | Type       | Formule / Règle | Explication |
        |---------------|-----------|-----------------|-------------|
        | 🆔 ID         | Calculé   | `next_id()`     | Identifiant unique auto si non fourni |
        | 📝 Name       | Manuel    | —               | Nom de la tâche |
        | 🏷️ Category  | Manuel    | Défaut `"Task"` | Catégorie de tâche |
        | 👤 Responsible| Manuel    | —               | Responsable de la tâche |
        | 🛠️ Equipment | Manuel    | —               | Ressources nécessaires |
        | 💬 Comments  | Manuel    | —               | Notes libres |
        | 🔗 Dependencies | Manuel | `"1,2"` → `[1,2]` | Parse automatique des prédécesseurs |

        ---

        ## 📅 2. Planification
        | Colonne       | Type       | Formule / Règle | Explication |
        |---------------|-----------|-----------------|-------------|
        | 🗓️ Start Date| Manuel/Calc | `latest_pred_end + 1j + lag` | Date de début (dépendances ou today) |
        | 🎯 End Date   | Calculé   | `start_date + duration_days - 1` | Date de fin prévue |
        | ⏳ Lag        | Manuel    | Défaut = 0 | Décalage manuel en jours |
        | 📊 Progress   | Manuel    | 0–100% | Avancement de la tâche |
        | 🔮 Projected End | Calculé | `start + (stochastic / speed)` | Fin projetée selon vitesse |
        | 🚀 Projection Speed | Calculé | `(stochastic × progress/100) / elapsed_days` | Vitesse réelle d’exécution |
        | ⚡ Critical   | Calculé   | `find_critical_path()` | Bool : tâche sur chemin critique |

        ---

        ## ⏱️ 3. Durées
        | Colonne         | Type     | Formule | Explication |
        |-----------------|----------|---------|-------------|
        | 😊 Optimistic   | Manuel   | — | Durée minimale attendue |
        | 😰 Pessimistic  | Manuel   | — | Durée maximale attendue |
        | 🤔 Probable     | Manuel/DB| `(opt + pess)/2` ou DB | Durée la plus probable |
        | 📈 Stochastic   | Calculé  | `(opt + 4*prob + pess)/6` | Durée PERT |
        | 📅 Duration(days)| Calculé | `round(stochastic)` | Durée finale arrondie |

        ---

        ## ⚠️ 4. Risques & Incertitudes
        | Colonne       | Type     | Formule | Explication |
        |---------------|----------|---------|-------------|
        | 📊 Std Dev    | Calculé  | `σ = (pess - opt) * sqrt(var_Y)` | Incertitude (Bêta-PERT ou approx `(pess-opt)/6`) |
        | 🛡️ Buffer     | Calculé  | Non-crit: `σ×{coeff_non}`<br>Critique: `√(σ_pred²+σ²)×{coeff_crit}`<br>Multi: `×{coeff_multi}` | Réserve de sécurité |
        | 🎲 P10–P90    | Calculé  | `beta.ppf(q, α, β)` → `opt + (pess-opt)×y_q` | Quantiles de durée |

        ---

        ✅ Les colonnes marquées *Manuel* doivent être saisies.  
        ⚙️ Les calculs sont appliqués automatiquement via `auto_calculate_all_tasks`.
        """)
    
     
    # === PAGE HEADER ===
    st.title("⚒️ Gestionnaire de Tâches Avancé")
    st.markdown("---")

    # === CONFIGURATION SIDEBAR ===
    st.sidebar.header("⚙️ Configuration des coefficients")
    
    # Buffer calculation coefficients
    tm.coefficient_non_critical = st.sidebar.number_input(
        "Coefficient non-critique", 
        value=float(tm.coefficient_non_critical), 
        min_value=0.1, 
        max_value=2.0, 
        step=0.1,
        help="Coefficient pour calculer le buffer des tâches non-critiques"
    )
    tm.coefficient_critical = st.sidebar.number_input(
        "Coefficient critique", 
        value=float(tm.coefficient_critical), 
        min_value=0.1, 
        max_value=2.0, 
        step=0.1,
        help="Coefficient pour calculer le buffer des tâches critiques"
    )
    tm.multiplier_multi_dependencies = st.sidebar.number_input(
        "Multiplicateur multi-dépendances", 
        value=float(tm.multiplier_multi_dependencies), 
        min_value=1.0, 
        max_value=3.0, 
        step=0.1,
        help="Multiplicateur supplémentaire pour les tâches avec plusieurs dépendances"
    )

    # === ACTION BUTTONS IN SIDEBAR ===
    st.sidebar.header("🛠️ Actions")

    if st.sidebar.button("🔄 Réinitialiser avec exemples", help="Charge des tâches d'exemple"):
        tm.create_sample()
        tm.save()
        st.rerun()

    if st.sidebar.button("💾 Sauvegarder", help="Sauvegarde manuelle"):
        tm.save()
        st.sidebar.success("✅ Sauvegardé!")

    # === DATA VALIDATION DISPLAY ===
    errors = tm.validate_task_data()
    if errors:
        st.sidebar.error("⚠️ Erreurs détectées:")
        for error in errors:
            st.sidebar.write(f"• {error}")

    # === MAIN INTERFACE LAYOUT ===
    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("📊 Tableau de Gestion des Tâches")

    with col2:
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

    # === INTERACTIVE DATA EDITOR ===
    edited_df = st.data_editor(
        pd.DataFrame(records),
        use_container_width=True,
        num_rows="dynamic",  # Allow adding/removing rows
        column_config=column_config,
        key="task_editor"
    )

    # === ACTION BUTTONS ===
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("🧮 **Calculer les champs manquants**", type="primary",
                     help="Calcule automatiquement les dates et durées manquantes"):
            with st.spinner("Calcul en cours..."):
                try:
                    update_tasks_from_editor(tm, edited_df)
                    tm.auto_calculate_all_tasks()
                    tm.save()

                    # 🔥 DEBUG 2 : Vérifie save
                    st.write("### ✅ DEBUG 2 - Save OK ?", f"Tâches: {len(tm.tasks)}")

                    # 🔥 DEBUG 3 : Écoute logs F12 pour erreurs
                    st.session_state.task_manager = tm
                    st.success("✅ Calculs effectués!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Debug 3 - Erreur: {e}")

    with col2:
        if st.button("💾 Appliquer les modifications", help="Applique les modifications du tableau"):
            try:
                update_tasks_from_editor(tm, edited_df)
                tm.save()
                st.session_state.task_manager = tm
                st.success("✅ Modifications appliquées!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Erreur lors de la sauvegarde: {e}")

    with col3:
        if st.button("🗑️ Supprimer sélection", help="Supprime les lignes sélectionnées"):
            st.info("Supprimez les lignes directement dans le tableau ci-dessus")

    with col4:
        if st.button("➕ Nouvelle tâche", help="Ajoute une nouvelle tâche vide"):
            new_id = tm.next_id()
            tm.tasks[new_id] = Task(id=new_id, name=f"Nouvelle tâche {new_id}")
            tm.save()
            st.session_state.task_manager = tm
            st.rerun()

    # === SUMMARY STATISTICS ===
    st.markdown("---")
    st.subheader("📋 Résumé Rapide")

    if tm.tasks:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_tasks = len(tm.tasks)
            st.metric("Total des tâches", total_tasks)

        with col2:
            completed_tasks = sum(1 for t in tm.tasks.values() if t.progress == 100)
            st.metric("Tâches terminées", completed_tasks)

        with col3:
            in_progress_tasks = sum(1 for t in tm.tasks.values() if 0 < t.progress < 100)
            st.metric("En cours", in_progress_tasks)

        with col4:
            critical_tasks = sum(1 for t in tm.tasks.values() if t.is_critical)
            st.metric("Tâches critiques", critical_tasks)

    # === HELP DOCUMENTATION ===
    with st.expander("ℹ️ Aide et Instructions"):
        st.markdown("""
        ### 📝 Comment utiliser ce gestionnaire de tâches :

        **Champs obligatoires :**
        - **Nom de la tâche** : Obligatoire pour chaque tâche
        - **Durée optimiste** et **Durée pessimiste** : Requis pour le calcul automatique PERT

        **Calculs automatiques :**
        - **Durée stochastique** : Calculée selon la formule PERT (O + 4M + P) / 6
        - **Écart type** : Calculé pour mesurer l'incertitude (P - O) / 6
        - **Buffer** : Calculé selon les règles de gestion des risques
        - **Dates** : Calculées selon les dépendances et durées
        - **État critique** : Déterminé par l'algorithme du chemin critique
        - **Date de fin projection** : Calculée si progression > 0

        **Format des dépendances :**
        - Séparez les IDs par des virgules : `1,2,3`
        - Ou par des points-virgules : `1;2;3`

        **Types de dépendances :**
        - **FS** (Finish-Start) : La tâche commence après la fin du prédécesseur
        - **SS** (Start-Start) : La tâche commence avec le prédécesseur
        - **FF** (Finish-Finish) : La tâche finit avec le prédécesseur
        - **SF** (Start-Finish) : La tâche finit quand le prédécesseur commence

        **Actions disponibles :**
        - **Calculer les champs manquants** : Lance tous les calculs automatiques
        - **Appliquer les modifications** : Sauvegarde vos changements sans recalculer
        - Ajoutez/supprimez des lignes directement dans le tableau

        **⚡ Fonctionnalités avancées :**
        - La date de fin projetée se calcule automatiquement si la progression > 0
        - Le calcul prend en compte la vitesse d'exécution réelle
        - Les buffers sont calculés différemment pour les tâches critiques et non-critiques
        - Les tâches avec multiples dépendances reçoivent un buffer supplémentaire
        """)

# ============================================================================================
# APPLICATION ENTRY POINT
# ============================================================================================

if __name__ == "__main__":
    main()
