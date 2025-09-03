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

import json
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Optional, List
import pandas as pd
import streamlit as st
import numpy as np

# ============================================================================================
# CONFIGURATION CONSTANTS
# ============================================================================================

# File paths for data persistence
SAVE_PATH = Path("data")  # Directory for saving project data
SAVE_FILE = SAVE_PATH / "mine_plan.json"  # Main project file

# ============================================================================================
# DATA MODEL - TASK DEFINITION
# ============================================================================================

@dataclass
class Task:
    """
    Core Task data class representing a single task in the mine planning project.
    
    This class uses the dataclass decorator for automatic generation of __init__, __repr__, etc.
    It contains all necessary fields for PERT analysis, critical path calculation, and progress tracking.
    """
    
    # === BASIC TASK INFORMATION ===
    id: int  # Unique task identifier
    name: str  # Task name/description
    category: str = "Task"  # Task category for organization
    responsible: Optional[str] = None  # Person/team responsible for the task
    equipment: Optional[str] = None  # Equipment required for the task
    comments: Optional[str] = None  # Additional notes/comments
    
    # === SCHEDULING INFORMATION ===
    start_date: Optional[date] = None  # Planned start date
    end_date: Optional[date] = None  # Planned end date
    projected_end_date: Optional[date] = None  # Projected end date based on current progress
    
    # === PERT DURATION ESTIMATES ===
    # PERT methodology uses three time estimates for each task
    duration_optimistic: Optional[int] = None  # Best-case scenario duration (days)
    duration_pessimistic: Optional[int] = None  # Worst-case scenario duration (days)
    duration_probable: Optional[int] = None  # Most likely duration (days)
    duration_stochastic: Optional[float] = None  # Calculated PERT duration
    duration_days: Optional[int] = None  # Final duration in days (rounded stochastic)
    
    # === STATISTICAL ANALYSIS ===
    standard_deviation: Optional[float] = None  # Statistical variance measure
    buffer: Optional[float] = None  # Risk buffer calculated based on task criticality
    
    # === DEPENDENCY MANAGEMENT ===
    dependencies: str = ""  # Comma-separated list of predecessor task IDs
    dependency_type: str = "FS"  # Dependency type: FS, SS, FF, SF
    lag: int = 0  # Lag time between dependent tasks (days)
    
    # === PROGRESS TRACKING ===
    progress: int = 0  # Task completion percentage (0-100)
    
    # === CRITICAL PATH ANALYSIS ===
    is_critical: Optional[bool] = None  # Whether task is on the critical path

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

    def get_expected_duration(self) -> Optional[float]:
        """
        Get the expected duration for the task.
        
        Priority order:
        1. Stochastic duration (PERT calculated)
        2. Duration in days (manually set)
        3. Simple PERT calculation if optimistic/pessimistic exist
        
        Returns:
            Optional[float]: Expected duration in days
        """
        if self.duration_stochastic is not None:
            return self.duration_stochastic
        elif self.duration_days is not None:
            return float(self.duration_days)
        elif self.duration_optimistic is not None and self.duration_pessimistic is not None:
            # Simple PERT calculation: (O + 4M + P) / 6
            probable = self.duration_probable or (self.duration_optimistic + self.duration_pessimistic) / 2
            return (self.duration_optimistic + 4 * probable + self.duration_pessimistic) / 6
        return None

    def get_actual_duration(self) -> Optional[int]:
        """
        Calculate actual duration if task is completed.
        
        Returns:
            Optional[int]: Actual duration in days if both dates exist
        """
        if self.start_date and self.end_date:
            return (self.end_date - self.start_date).days + 1
        return None

    def get_remaining_duration(self) -> Optional[float]:
        """
        Calculate remaining duration based on current progress.
        
        Returns:
            Optional[float]: Remaining duration in days
        """
        expected = self.get_expected_duration()
        if expected and self.progress < 100:
            return expected * (100 - self.progress) / 100
        return 0 if self.progress == 100 else expected

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

    def load(self) -> bool:
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

            # Create Task object with all stored data
            self.tasks[int(k)] = Task(
                id=int(k),
                name=d["name"],
                start_date=date.fromisoformat(d.get("start_date")) if d.get("start_date") else None,
                end_date=date.fromisoformat(d.get("end_date")) if d.get("end_date") else None,
                duration_optimistic=d.get("duration_optimistic"),
                duration_pessimistic=d.get("duration_pessimistic"),
                duration_probable=d.get("duration_probable"),
                duration_stochastic=d.get("duration_stochastic"),
                responsible=d.get("responsible"),
                equipment=d.get("equipment"),
                category=d.get("category", "Task"),
                dependencies=dependencies,
                dependency_type=d.get("dependency_type", "FS"),
                lag=d.get("lag", 0),
                comments=d.get("comments"),
                progress=d.get("progress", 0),
                duration_days=d.get("duration_days"),
                standard_deviation=d.get("standard_deviation"),
                buffer=d.get("buffer"),
                projected_end_date=date.fromisoformat(d.get("projected_end_date")) if d.get("projected_end_date") else None,
                is_critical=d.get("is_critical")
            )
        return True

    def create_sample(self):
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
            1: Task(1, "Préparation du site", td, None, 5, 10, 7, None, "Équipe A", "Bulldozer", progress=30),
            2: Task(2, "Construction route d'accès", None, None, 7, 12, 9, None, "Équipe B", "Excavatrice", dependencies="1", progress=15),
            3: Task(3, "Forage initial", None, None, 10, 18, 14, None, "Équipe C", "Foreuse", dependencies="1", progress=50),
            4: Task(4, "Installation équipements", None, None, 3, 8, 5, None, "Équipe D", "Grue", dependencies="2,3", progress=0),
        }

    # ========================================================================================
    # PERT CALCULATION METHODS
    # ========================================================================================

    def calculate_stochastic_duration(self, task: Task) -> Optional[float]:
        """
        Calculate PERT stochastic duration using the beta distribution formula.
        
        PERT Formula: (Optimistic + 4 × Most Likely + Pessimistic) / 6
        
        Args:
            task (Task): Task to calculate duration for
            
        Returns:
            Optional[float]: Calculated stochastic duration
        """
        if task.duration_optimistic is None or task.duration_pessimistic is None:
            return task.duration_stochastic  # Keep existing value if no input data
            
        # Use provided probable duration or calculate average
        if task.duration_probable is None:
            task.duration_probable = (task.duration_optimistic + task.duration_pessimistic) / 2
        
        # Apply PERT formula
        task.duration_stochastic = (
            task.duration_optimistic + 4 * task.duration_probable + task.duration_pessimistic
        ) / 6
        
        return task.duration_stochastic

    def calculate_standard_deviation(self, task: Task) -> Optional[float]:
        """
        Calculate standard deviation for task duration.
        
        Formula: (Pessimistic - Optimistic) / 6
        This represents the uncertainty in task duration estimates.
        
        Args:
            task (Task): Task to calculate standard deviation for
            
        Returns:
            Optional[float]: Calculated standard deviation
        """
        if task.duration_optimistic is None or task.duration_pessimistic is None:
            return task.standard_deviation  # Keep existing value
            
        task.standard_deviation = (task.duration_pessimistic - task.duration_optimistic) / 6
        return task.standard_deviation

    def calculate_buffer(self, task: Task, predecessor_std: Optional[float] = None) -> Optional[float]:
        """
        Calculate risk buffer for task based on criticality and dependencies.
        
        Buffer calculation rules:
        - Critical tasks: Use higher coefficient and combine with predecessor variance
        - Non-critical tasks: Use lower coefficient
        - Multi-dependency tasks: Apply additional multiplier
        
        Args:
            task (Task): Task to calculate buffer for
            predecessor_std (Optional[float]): Standard deviation from predecessor tasks
            
        Returns:
            Optional[float]: Calculated buffer in days
        """
        if task.duration_optimistic is None or task.duration_pessimistic is None:
            return task.buffer  # Keep existing value
            
        # Ensure standard deviation is calculated
        if task.standard_deviation is None:
            self.calculate_standard_deviation(task)
            
        if task.standard_deviation is None:
            return task.buffer
        
        # Analyze task characteristics
        num_predecessors = len(task.get_dependency_ids())
        is_multi_dependency = num_predecessors > 1
        
        if task.is_critical:
            # Critical task buffer calculation
            if predecessor_std is not None:
                # Combine variances using root sum of squares
                combined_std = np.sqrt(predecessor_std**2 + task.standard_deviation**2)
            else:
                combined_std = task.standard_deviation
            task.buffer = self.coefficient_critical * combined_std
        else:
            # Non-critical task buffer calculation
            task.buffer = self.coefficient_non_critical * task.standard_deviation
        
        # Apply multiplier for tasks with multiple dependencies
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
        
        # === PHASE 1: PERT CALCULATIONS ===
        # Calculate stochastic durations and standard deviations first
        # These are fundamental metrics needed for all other calculations
        for task in self.tasks.values():
            if task.duration_optimistic is not None and task.duration_pessimistic is not None:
                # Only calculate if not already computed (preserves manual overrides)
                if task.duration_stochastic is None:
                    self.calculate_stochastic_duration(task)
                if task.standard_deviation is None:
                    self.calculate_standard_deviation(task)
                # Set integer duration for scheduling calculations
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
        # Calculate projected completion dates based on current progress
        for task in self.tasks.values():
            # Always recalculate projections if there's progress
            if task.progress > 0 and task.start_date and task.duration_stochastic:
                self.calculate_projected_end_date(task)

        # === PHASE 6: BUFFER INTEGRATION ===
        # Integrate calculated buffers into schedule durations
        for task in self.tasks.values():
            if (task.start_date and task.duration_stochastic is not None and 
                task.buffer is not None and task.duration_days):
                # Calculate total duration including buffer
                total_duration = int(round(task.duration_stochastic + task.buffer))
                if total_duration != task.duration_days:
                    task.duration_days = total_duration
                    # Update end date to reflect buffered duration
                    if task.start_date:
                        task.end_date = task.start_date + timedelta(days=task.duration_days - 1)

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

def update_tasks_from_editor(tm: TaskManager, edited_df: pd.DataFrame):
    """
    Update TaskManager from Streamlit data editor DataFrame.
    
    This function bridges the gap between the Streamlit UI and the TaskManager
    business logic. It handles data type conversions, validation, and preservation
    of calculated values.
    
    Args:
        tm (TaskManager): TaskManager instance to update
        edited_df (pd.DataFrame): DataFrame from Streamlit data editor
    """
    
    # === IDENTIFY TASK CHANGES ===
    # Determine which tasks to keep, update, or delete
    new_ids = set()
    for _, row in edited_df.iterrows():
        if pd.notna(row.get("ID")):
            new_ids.add(int(row["ID"]))

    # Remove deleted tasks
    for tid in list(tm.tasks.keys()):
        if tid not in new_ids:
            del tm.tasks[tid]

    # === PROCESS EACH ROW FROM EDITOR ===
    for _, row in edited_df.iterrows():
        # Skip rows without task names
        if pd.isna(row.get("Nom de la tâche*")) or not str(row.get("Nom de la tâche*", "")).strip():
            continue

        # Extract basic task information
        name = str(row.get("Nom de la tâche*", "")).strip()
        tid = int(row["ID"]) if pd.notna(row.get("ID")) else tm.next_id()

        # === DATE PARSING UTILITIES ===
        def parse_date(date_str):
            """
            Robust date parsing for various input formats.
            
            Args:
                date_str: Date string from UI
                
            Returns:
                Optional[date]: Parsed date or None
            """
            if pd.isna(date_str) or not str(date_str).strip():
                return None
            try:
                date_str = str(date_str).strip()
                if len(date_str) == 10 and '-' in date_str:
                    return date.fromisoformat(date_str)
            except (ValueError, TypeError):
                pass
            return None

        # Parse all date fields
        start_date = parse_date(row.get("Date début"))
        end_date = parse_date(row.get("Date fin"))
        projected_end_date = parse_date(row.get("Date de fin projection"))

        # === NUMERIC PARSING UTILITIES ===
        def safe_int(value):
            """Safe integer conversion with None handling."""
            if pd.isna(value) or value == "":
                return None
            try:
                return int(float(value))  # Handle float strings
            except (ValueError, TypeError):
                return None
            
        def safe_float(value):
            """Safe float conversion with None handling."""
            if pd.isna(value) or value == "":
                return None
            try:
                return float(value)
            except (ValueError, TypeError):
                return None

        # === PRESERVE CALCULATED VALUES ===
        # Get existing task to preserve calculated fields
        existing_task = tm.tasks.get(tid)
        
        # === CREATE/UPDATE TASK ===
        task = Task(
            id=tid,
            name=name,
            responsible=str(row.get("Responsable", "")).strip() or None,
            equipment=str(row.get("Équipements", "")).strip() or None,
            start_date=start_date,
            end_date=end_date,
            duration_optimistic=safe_int(row.get("Durée optimiste")),
            duration_pessimistic=safe_int(row.get("Durée pessimiste")),
            duration_probable=safe_int(row.get("Durée probable")),
            # Preserve calculated stochastic duration unless manually overridden
            duration_stochastic=safe_float(row.get("Durée stochastique")) if existing_task is None else (safe_float(row.get("Durée stochastique")) or existing_task.duration_stochastic),
            duration_days=safe_int(row.get("Durée (jours)")),
            dependencies=str(row.get("Dépendance", "")).strip(),
            dependency_type=str(row.get("Type Dép.", "FS")).strip(),
            lag=safe_int(row.get("Décalage (j)")) or 0,
            progress=safe_int(row.get("Progression")) or 0,
            comments=str(row.get("Commentaires", "")).strip() or None,
            # Preserve calculated fields
            standard_deviation=safe_float(row.get("Écart type")) if existing_task is None else (safe_float(row.get("Écart type")) or existing_task.standard_deviation),
            buffer=safe_float(row.get("Buffer")) if existing_task is None else (safe_float(row.get("Buffer")) or existing_task.buffer),
            projected_end_date=projected_end_date if existing_task is None else (projected_end_date or existing_task.projected_end_date),
            is_critical=bool(row.get("État critique")) if pd.notna(row.get("État critique")) else (existing_task.is_critical if existing_task else None),
        )

        tm.tasks[tid] = task

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
        ["📊 Gestion des Tâches", "📈 Visualisations & Statistiques"]
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
            from visualizations import show_visualizations_page
            show_visualizations_page(tm)
        except Exception as e:
            st.error("❌ Erreur lors du chargement des visualisations :")
            st.code(str(e))
    else:
        st.warning("Page inconnue")

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
        records.append({
            "ID": task.id,
            "Nom de la tâche*": task.name,
            "Responsable": task.responsible or "",
            "Équipements": task.equipment or "",
            "Date début": task.start_date.strftime("%Y-%m-%d") if task.start_date else "",
            "Date fin": task.end_date.strftime("%Y-%m-%d") if task.end_date else "",
            "Durée optimiste": task.duration_optimistic,
            "Durée pessimiste": task.duration_pessimistic,
            "Durée probable": task.duration_probable,
            "Durée stochastique": round(task.duration_stochastic, 2) if task.duration_stochastic is not None else None,
            "Dépendance": task.dependencies,
            "Durée (jours)": task.duration_days,
            "Progression": task.progress,
            "Date de fin projection": task.projected_end_date.strftime("%Y-%m-%d") if task.projected_end_date else "",
            "État critique": task.is_critical,
            "Écart type": round(task.standard_deviation, 2) if task.standard_deviation is not None else None,
            "Buffer": round(task.buffer, 2) if task.buffer is not None else None,
            "Commentaires": task.comments or "",
        })

    # Add empty row if no tasks exist
    if not records:
        records.append({
            "ID": None,
            "Nom de la tâche*": "",
            "Responsable": "",
            "Équipements": "",
            "Date début": "",
            "Date fin": "",
            "Durée optimiste": None,
            "Durée pessimiste": None,
            "Durée probable": None,
            "Durée stochastique": None,
            "Dépendance": "",
            "Durée (jours)": None,
            "Progression": 0,
            "Date de fin projection": "",
            "État critique": None,
            "Écart type": None,
            "Buffer": None,
            "Commentaires": "",
        })

    # === COLUMN CONFIGURATION ===
    # Define how each column should behave in the data editor
    column_config = {
        "ID": st.column_config.NumberColumn("ID", disabled=True, width=50),
        "Nom de la tâche*": st.column_config.TextColumn("Nom de la tâche*", required=True, width=150),
        "Responsable": st.column_config.TextColumn("Responsable", width=120),
        "Équipements": st.column_config.TextColumn("Équipements", width=120),
        "Date début": st.column_config.TextColumn("Date début", help="Format: YYYY-MM-DD", width=100),
        "Date fin": st.column_config.TextColumn("Date fin", help="Format: YYYY-MM-DD", width=100),
        "Durée optimiste": st.column_config.NumberColumn("Durée optimiste", min_value=0, width=80),
        "Durée pessimiste": st.column_config.NumberColumn("Durée pessimiste", min_value=0, width=80),
        "Durée probable": st.column_config.NumberColumn("Durée probable", min_value=0, width=80),
        "Durée stochastique": st.column_config.NumberColumn("Durée stochastique", disabled=True, width=80),
        "Dépendance": st.column_config.TextColumn("Dépendance", help="Ex: 1,2,3", width=100),
        "Durée (jours)": st.column_config.NumberColumn("Durée (j)", min_value=0, width=80),
        "Progression": st.column_config.NumberColumn("Progression", min_value=0, max_value=100, step=5, width=100),
        "Date de fin projection": st.column_config.TextColumn("Date de fin projection", disabled=True, width=100),
        "État critique": st.column_config.CheckboxColumn("État critique", disabled=True, width=80),
        "Écart type": st.column_config.NumberColumn("Écart type", disabled=True, width=80),
        "Buffer": st.column_config.NumberColumn("Buffer", disabled=True, width=80),
        "Commentaires": st.column_config.TextColumn("Commentaires", width=200),
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
                    # Update tasks from editor data
                    update_tasks_from_editor(tm, edited_df)
                    
                    # Run comprehensive calculations
                    tm.auto_calculate_all_tasks()
                    
                    # Save results
                    tm.save()
                    
                    # Update session state
                    st.session_state.task_manager = tm
                    
                    st.success("✅ Calculs effectués avec succès!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Erreur lors du calcul: {e}")

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
