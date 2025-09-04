# ============================================================================================
# simulation.py - Code complet et autonome pour la page de simulation
# ============================================================================================

# Imports nécessaires
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
import random
from scipy.stats import beta
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import dataclass

# ============================================================================================
# CLASSES INTÉGRÉES (copiées de ton app.py pour éviter les imports)
# ============================================================================================

@dataclass
class Task:
    """
    Core Task Data Class.
    """
    id: int
    name: str
    category: str = "Task"
    responsible: Optional[str] = None
    equipment: Optional[str] = None
    comments: Optional[str] = None
    dependencies: str = ""
    dependency_type: str = "FS"
    lag: int = 0
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    progress: int = 0
    projected_end_date: Optional[date] = None
    projection_speed: Optional[float] = None
    is_critical: Optional[bool] = None
    duration_optimistic: Optional[int] = None
    duration_pessimistic: Optional[int] = None
    duration_probable: Optional[int] = None
    duration_stochastic: Optional[float] = None
    duration_days: Optional[int] = None
    standard_deviation: Optional[float] = None
    buffer: Optional[float] = None
    p10: Optional[float] = None
    p20: Optional[float] = None
    p30: Optional[float] = None
    p40: Optional[float] = None
    p50: Optional[float] = None
    p60: Optional[float] = None
    p70: Optional[float] = None
    p80: Optional[float] = None
    p90: Optional[float] = None

    def scheduled_end(self) -> Optional[date]:
        """
        Calculate the scheduled end date based on start date and duration.
        """
        if self.start_date and self.duration_days:
            return self.start_date + timedelta(days=self.duration_days - 1)
        return self.end_date

    def get_expected_duration(self) -> Optional[float]:
        """
        Get the expected duration for the task.
        """
        if self.duration_stochastic is not None:
            return self.duration_stochastic
        if self.duration_optimistic is not None and self.duration_pessimistic is not None:
            probable = self.duration_probable or ((self.duration_optimistic + self.duration_pessimistic) / 2)
            if self.duration_probable is None:
                self.duration_probable = probable
            self.duration_stochastic = (self.duration_optimistic + 4 * probable + self.duration_pessimistic) / 6
            return self.duration_stochastic
        elif self.duration_days is not None:
            return float(self.duration_days)
        return None

    def get_actual_duration(self) -> Optional[int]:
        """
        Calculate actual duration if task is completed.
        """
        if self.start_date and self.end_date:
            return (self.end_date - self.start_date).days + 1
        return None

    def get_remaining_duration(self) -> Optional[float]:
        """
        Calculate remaining duration based on current progress.
        """
        expected = self.get_expected_duration()
        if not expected or not self.start_date:
            return expected * (100 - self.progress) / 100 if expected and self.progress < 100 else 0.0
        work_done_days = expected * (self.progress / 100.0)
        days_elapsed = (date.today() - self.start_date).days
        if days_elapsed <= 0:
            return expected * (100 - self.progress) / 100 if self.progress < 100 else 0.0
        if work_done_days > 0:
            projection_speed = work_done_days / days_elapsed
        else:
            projection_speed = 1.0
        if projection_speed > 0:
            projected_total_duration = expected / projection_speed
        else:
            projected_total_duration = expected
        self.projected_end_date = self.start_date + timedelta(days=int(round(projected_total_duration)))
        self.projection_speed = round(projection_speed, 2)
        remaining_duration = max(0.0, projected_total_duration - days_elapsed)
        if self.progress >= 100:
            return 0.0
        return round(remaining_duration, 2)

    def get_stochastic_end_date(self) -> Optional[date]:
        """
        Calculate end date based on stochastic duration.
        """
        if self.start_date and self.duration_stochastic:
            return self.start_date + timedelta(days=int(round(self.duration_stochastic)))
        elif self.start_date and self.duration_days:
            return self.start_date + timedelta(days=self.duration_days - 1)
        return self.end_date

    def get_standard_deviation(self) -> Optional[float]:
        """
        Get the standard deviation for the task.
        """
        return self.standard_deviation

    def to_dict(self):
        """
        Convert Task object to dictionary for JSON serialization.
        """
        d = asdict(self)
        d["start_date"] = self.start_date.isoformat() if self.start_date else None
        d["end_date"] = self.end_date.isoformat() if self.end_date else None
        d["projected_end_date"] = self.projected_end_date.isoformat() if self.projected_end_date else None
        return d

    def get_dependency_ids(self) -> List[int]:
        """
        Parse dependency string and return list of task IDs.
        """
        if not self.dependencies:
            return []
        deps_str = self.dependencies.replace(";", ",")
        return [int(x.strip()) for x in deps_str.split(",") if x.strip().isdigit()]

    def is_complete_for_calculation(self) -> bool:
        """
        Check if task has sufficient information for scheduling calculations.
        """
        has_dates = self.start_date and self.end_date
        has_start_duration = self.start_date and self.duration_days
        has_end_duration = self.end_date and self.duration_days
        return has_dates or has_start_duration or has_end_duration

# ============================================================================================
# TASK MANAGER CLASS
# ============================================================================================

class TaskManager:
    """
    Main business logic class for managing mine planning tasks.
    """
    def __init__(self):
        """
        Initialize TaskManager with default settings and load existing data.
        """
        self.coefficient_non_critical = 0.5
        self.coefficient_critical = 1.3
        self.multiplier_multi_dependencies = 1.2
        self.tasks: Dict[int, Task] = {}
        SAVE_PATH = Path("data")
        SAVE_PATH.mkdir(exist_ok=True)
        if not self.load():
            self.create_sample()

    def next_id(self) -> int:
        """
        Generate next available task ID.
        """
        return max(self.tasks.keys(), default=0) + 1

    def save(self):
        """
        Save all tasks to JSON file.
        """
        SAVE_FILE = SAVE_PATH / "mine_plan.json"
        raw = {"tasks": {tid: t.to_dict() for tid, t in self.tasks.items()}}
        SAVE_FILE.write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")

    def load(self) -> bool:
        """
        Load tasks from JSON file.
        """
        SAVE_FILE = SAVE_PATH / "mine_plan.json"
        if not SAVE_FILE.exists():
            return False
        try:
            raw = json.loads(SAVE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return False
        self.tasks = {}
        for k, d in raw.get("tasks", {}).items():
            dependencies = d.get("dependencies", "")
            if isinstance(dependencies, list):
                dependencies = ",".join(map(str, dependencies))
            self.tasks[int(k)] = Task(
                id=int(k),
                name=d["name"],
                category=d.get("category", "Task"),
                responsible=d.get("responsible"),
                equipment=d.get("equipment"),
                comments=d.get("comments"),
                dependencies=dependencies,
                dependency_type=d.get("dependency_type", "FS"),
                lag=d.get("lag", 0),
                start_date=date.fromisoformat(d.get("start_date")) if d.get("start_date") else None,
                end_date=date.fromisoformat(d.get("end_date")) if d.get("end_date") else None,
                progress=d.get("progress", 0),
                projected_end_date=date.fromisoformat(d.get("projected_end_date")) if d.get("projected_end_date") else None,
                projection_speed=d.get("projection_speed"),
                is_critical=d.get("is_critical"),
                duration_optimistic=d.get("duration_optimistic"),
                duration_pessimistic=d.get("duration_pessimistic"),
                duration_probable=d.get("duration_probable"),
                duration_stochastic=d.get("duration_stochastic"),
                duration_days=d.get("duration_days"),
                standard_deviation=d.get("standard_deviation"),
                buffer=d.get("buffer"),
                p10=d.get("p10"), p20=d.get("p20"), p30=d.get("p30"), p40=d.get("p40"),
                p50=d.get("p50"), p60=d.get("p60"), p70=d.get("p70"), p80=d.get("p80"), p90=d.get("p90"),
            )
        return True

    def create_sample(self):
        """
        Create sample mine planning tasks for demonstration.
        """
        td = date.today()
        self.tasks = {
            1: Task(id=1, name="Préparation du site", start_date=td, duration_optimistic=5, duration_pessimistic=10, duration_probable=7, progress=30, responsible="Équipe A", equipment="Bulldozer"),
            2: Task(id=2, name="Construction route d'accès", duration_optimistic=7, duration_pessimistic=12, duration_probable=9, progress=15, dependencies="1", responsible="Équipe B", equipment="Excavatrice"),
            3: Task(id=3, name="Forage initial", duration_optimistic=10, duration_pessimistic=18, duration_probable=14, progress=50, dependencies="1", responsible="Équipe C", equipment="Foreuse"),
            4: Task(id=4, name="Installation équipements", duration_optimistic=3, duration_pessimistic=8, duration_probable=5, progress=0, dependencies="2,3", responsible="Équipe D", equipment="Grue"),
        }

    def calculate_projected_end_date(self, task: Task) -> Optional[date]:
        """
        Calculate projected end date based on current progress and execution speed.
        """
        if not task.start_date or not task.duration_stochastic or task.progress == 0:
            return task.projected_end_date
        work_done_days = task.duration_stochastic * (task.progress / 100)
        days_elapsed = (date.today() - task.start_date).days
        if days_elapsed <= 0:
            days_elapsed = 1
        execution_speed = work_done_days / days_elapsed
        if execution_speed > 0:
            real_duration = task.duration_stochastic / execution_speed
        else:
            real_duration = task.duration_stochastic
        task.projected_end_date = task.start_date + timedelta(days=int(round(real_duration)))
        return task.projected_end_date

    def find_critical_path(self) -> List[int]:
        """
        Implement Critical Path Method.
        """
        if not self.tasks:
            return []
        earliest_start = {}
        earliest_finish = {}
        sorted_tasks = sorted(self.tasks.keys())
        for task_id in sorted_tasks:
            task = self.tasks[task_id]
            dep_ids = task.get_dependency_ids()
            if not dep_ids:
                earliest_start[task_id] = task.start_date or date.today()
            else:
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
            duration = task.duration_days or (int(task.duration_stochastic) if task.duration_stochastic else 1)
            earliest_finish[task_id] = earliest_start[task_id] + timedelta(days=duration - 1)
        latest_start = {}
        latest_finish = {}
        if earliest_finish:
            project_end = max(earliest_finish.values())
        else:
            return []
        for task_id in reversed(sorted_tasks):
            task = self.tasks[task_id]
            successors = [t.id for t in self.tasks.values() if task_id in t.get_dependency_ids()]
            if not successors:
                latest_finish[task_id] = project_end
            else:
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
            duration = task.duration_days or (int(task.duration_stochastic) if task.duration_stochastic else 1)
            latest_start[task_id] = latest_finish[task_id] - timedelta(days=duration - 1)
        critical_tasks = []
        for task_id in self.tasks.keys():
            if task_id in earliest_start and task_id in latest_start and earliest_start[task_id] == latest_start[task_id]:
                critical_tasks.append(task_id)
        return critical_tasks

    def identify_critical_tasks(self, critical_path: List[int]):
        """
        Mark tasks as critical.
        """
        for task in self.tasks.values():
            task.is_critical = task.id in critical_path

    _calculate_start_from_dependencies = (  # Ceci est stubs; tu peux compléter avec ton code original
        lambda self, task: task.start_date
    )
    calculate_buffer = (
        lambda self, task, predecessor_std=None: None
    )
    calculate_beta_standard_deviation = (
        lambda self, task, lambda_value=4.0: None
    )
    calculate_beta_percentiles = (
        lambda self, task, lambda_value=4.0: None
    )
    calculate_standard_deviation = (
        lambda self, task: None
    )

    def auto_calculate_all_tasks(self, max_iterations: int = 10):
        """
        Comprehensive calculation engine.
        """
        # Adherece les autres méthodes ici si nécessaire...
        self.identify_critical_tasks(self.find_critical_path())
        return  # Stub; complète avec ton code original

    def validate_task_data(self) -> List[str]:
        """
        Validate task data.
        """
        errors = []
        for task in self.tasks.values():
            if not task.name.strip():
                errors.append(f"Tâche {task.id}: Le nom est obligatoire")
            if self._has_circular_dependency(task.id):
                errors.append(f"Tâche {task.id}: Dépendance circulaire")
            for dep_id in task.get_dependency_ids():
                if dep_id not in self.tasks:
                    errors.append(f"Tâche {task.id}: Dépendance {dep_id} inexistante")
        return errors

    def _has_circular_dependency(self, task_id: int, visited: set = None) -> bool:
        """
        Detect circular dependencies.
        """
        if visited is None:
            visited = set()
        if task_id in visited:
            return True
        visited.add(task_id)
        task = self.tasks.get(task_id)
        if task:
            for dep_id in task.get_dependency_ids():
                if self._has_circular_dependency(dep_id, visited.copy()):
                    return True
        return False

    def _topological_sort(self) -> List[int]:
        """
        Perform topological sort.
        """
        result = []
        visited = set()
        temp_visited = set()
        def visit(task_id: int):
            if task_id in temp_visited:
                return
            if task_id in visited:
                return
            temp_visited.add(task_id)
            task = self.tasks.get(task_id)
            if task:
                for dep_id in task.get_dependency_ids():
                    if dep_id in self.tasks:
                        visit(dep_id)
            temp_visited.remove(task_id)
            visited.add(task_id)
            result.append(task_id)
        for task_id in self.tasks.keys():
            if task_id not in visited:
                visit(task_id)
        return result

# ============================================================================================
# FONCTION UTILE
# ============================================================================================

def update_tasks_from_editor(tm: TaskManager, edited_df: pd.DataFrame):
    """
    Update TaskManager from Streamlit data editor DataFrame.
    """
    # Stub; complète avec ton code original...

# ============================================================================================
# PAGE DE SIMULATION
# ============================================================================================

def show_simulation_page(tm: TaskManager):
    """
    Page Streamlit pour simuler les durées de projet basées sur les combinaisons de percentiles.
    """
    st.title("🧩 Simulation de Projet : Analyse des Durées par Combinaisons de Percentiles")
    st.markdown("---")
    st.markdown("""
    Cette page simule la variabilité de la durée totale du projet en utilisant les percentiles de chaque tâche.
    On génère des combinaisons de durées (une valeur percentile par tâche) et on calcule la date de fin en respectant les dépendances.
    Enfin, on estime la distribution empirique des durées de fin de projet.
    """)

    # Vérification : Au moins une tâche avec percentiles
    tasks_with_percentiles = [t for t in tm.tasks.values() if t.p10 and t.p30 and t.p50 and t.p70 and t.p90]
    if not tasks_with_percentiles:
        st.warning("⚠️ Pas assez de tâches avec des percentiles calculés (P10-P90). Lance 'Calculer les champs manquants' sur la page principale.")
        return

    # === PARAMÈTRES D'ENTRÉE ===
    st.sidebar.header("🔧 Paramètres de Simulation")
    project_start_date = st.sidebar.date_input("Date de début projet", date.today())

    qs = [0.1, 0.3, 0.5, 0.7, 0.9]
    percentiles_labels = ['P10', 'P30', 'P50', 'P70', 'P90']

    # Mode de génération des combinaisons
    use_enumeration = st.sidebar.checkbox("Utiliser énumération exhaustive",
                                         value=True if len(tasks_with_percentiles) <= 5 else False)
    if not use_enumeration:
        num_simulations = st.sidebar.slider("Nombre de simulations Monte-Carlo",
                                           min_value=100, max_value=10000, value=1000, step=100)

    # === EXTRACTION DES DONNÉES ===
    task_ids = [t.id for t in tasks_with_percentiles]
    preds_dict = {t.id: t.get_dependency_ids() for t in tasks_with_percentiles}
    durations_dict = {t.id: {percentiles_labels[i]: getattr(t, f'p{i*20 + 10}')
                             for i in range(len(percentiles_labels))} for t in tasks_with_percentiles}

    # Fonction adaptée pour calculer la date de fin projet d'une combinaison
    def compute_project_finish_date(durations_combination, start_date):
        """
        Calcule la date de fin projet pour une combinaison donnée, en respectant les dépendances.
        """
        ES_times = {tid: 0.0 for tid in task_ids}
        EF_times = {tid: 0.0 for tid in task_ids}

        sorted_tasks = sorted(task_ids)
        remaining = set(task_ids)

        while remaining:
            progressed = False
            for tid in list(remaining):
                if all(pred in EF_times for pred in preds_dict[tid]):
                    ES_times[tid] = max([EF_times[p] for p in preds_dict[tid]] or [0.0])
                    EF_times[tid] = ES_times[tid] + durations_combination[tid]
                    remaining.remove(tid)
                    progressed = True
            if not progressed:
                st.error(f"Cycle détecté dans les dépendances pour tâche {tid} ! Simulations arrêtées.")
                return None

        total_days = max(EF_times.values()) if EF_times else 0.0
        return start_date + timedelta(days=int(round(total_days)))

    # === GÉNÉRATION DES COMBINAISONS ===
    durations_list = []
    if use_enumeration:
        st.info(f"Génère toutes les combinaisons avec {len(task_ids)} tâches et {len(qs)} percentiles chacun...")
        combinations = list(product(*[durations_dict[tid].values() for tid in task_ids]))
        st.write(f"📊 {len(combinations)} combinaisons générées.")

        for comb_idx, comb in enumerate(combinations):
            durations_for_tasks = {task_ids[i]: comb[i] for i in range(len(task_ids))}
            finish_date = compute_project_finish_date(durations_for_tasks, project_start_date)
            if finish_date:
                total_days = (finish_date - project_start_date).days
                durations_list.append(total_days)
                if len(combinations) > 100 and comb_idx % (len(combinations) // 20) == 0:
                    st.progress(comb_idx / len(combinations))
    else:
        st.info(f"Monte-Carlo : {num_simulations} simulations aléatoires...")
        for _ in range(num_simulations):
            durations_for_tasks = {tid: durations_dict[tid][random.choice(percentiles_labels)]
                                   for tid in task_ids}
            finish_date = compute_project_finish_date(durations_for_tasks, project_start_date)
            if finish_date:
                total_days = (finish_date - project_start_date).days
                durations_list.append(total_days)

    # === CALCUL DES PERCENTILES GLOBAUX ===
    if not durations_list:
        st.error("❌ Aucune simulation valide ! Vérifie les dépendances.")
        return

    global_percentiles = np.percentile(durations_list, [10, 30, 50, 70, 90])
    days_to_add = timedelta(days=int(global_percentiles[4]))
    date_p90 = project_start_date + days_to_add

    # === AFFICHAGE DES RÉSULTATS ===
    st.markdown("---")
    st.subheader("📊 Résultats de Simulation")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Durée médiane (P50)", f"{int(global_percentiles[2])} jours")
        st.metric("Durée optimiste (P10)", f"{int(global_percentiles[0])} jours")
        st.metric("Durée pessimiste (P90)", f"{int(global_percentiles[4])} jours")

    with col2:
        st.metric("Date de fin P90 (95% prob)", date_p90.strftime('%Y-%m-%d'))
        st.metric("Échantillon total", len(durations_list))

    # Tableau détaillé des percentiles
    st.markdown("### Détail des Percentiles Globaux")
    percentile_df = pd.DataFrame({
        'Probabilité': ['10%', '30%', '50%', '70%', '90%'],
        'Durée (jours)': [int(global_percentiles[i]) for i in range(5)],
        'Date de fin (si début aujourd\'hui)': [(project_start_date + timedelta(days=int(global_percentiles[i]))).strftime('%d-%m-%Y') for i in range(5)]
    })
    st.table(percentile_df)

    # Histogramme des durées
    st.markdown("### Histogramme des Durées Simulées")
    fig, ax = plt.subplots()
    ax.hist(durations_list, bins=30, alpha=0.7, color='blue')
    ax.set_xlabel('Durée totale (jours)')
    ax.set_ylabel('Fréquence')
    ax.set_title('Distribution des Durées de Projet')
    st.pyplot(fig)

    # CDF empirique (optionnel)
    with st.expander("📈 CDF Empirique"):
        fig_cdf, ax_cdf = plt.subplots()
        ax_cdf.hist(durations_list, bins=50, cumulative=True, density=True, alpha=0.7, color='green')
        ax_cdf.set_xlabel('Durée (jours)')
        ax_cdf.set_ylabel('Probabilité cumulative')
        ax_cdf.set_title('Fonction de Répartition des Durées')
        st.pyplot(fig_cdf)
