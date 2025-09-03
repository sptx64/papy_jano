import json
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Optional, List
import pandas as pd
import streamlit as st
import numpy as np


# Memory file 
SAVE_PATH = Path("data")
SAVE_FILE = SAVE_PATH / "mine_plan.json"

# ---------------------------- Model ----------------------------
# Tableau 
@dataclass
class Task:
    id: int
    name: str
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    duration_optimistic: Optional[int] = None
    duration_pessimistic: Optional[int] = None
    duration_probable: Optional[int] = None
    duration_stochastic: Optional[float] = None
    responsible: Optional[str] = None
    equipment: Optional[str] = None
    category: str = "Task"
    dependencies: str = ""  # Format: "1,2,3" ou "1;2;3"
    dependency_type: str = "FS"  # FS, SS, FF, SF
    lag: int = 0
    comments: Optional[str] = None
    progress: int = 0
    duration_days: Optional[int] = None
    standard_deviation: Optional[float] = None
    buffer: Optional[float] = None
    projected_end_date: Optional[date] = None
    is_critical: Optional[bool] = None

    class TaskManager:
    def __init__(self):
        SAVE_PATH.mkdir(exist_ok=True)
        self.tasks: Dict[int, Task] = {}
        self.coefficient_non_critical = 0.5
        self.coefficient_critical = 1.3
        self.multiplier_multi_dependencies = 1.2
        self.load() or self.create_sample()

    def next_id(self) -> int:
        return max(self.tasks.keys(), default=0) + 1

    def save(self):
        raw = {"tasks": {tid: t.to_dict() for tid, t in self.tasks.items()}}
        SAVE_FILE.write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")

    def load(self) -> bool:
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
        td = date.today()
        self.tasks = {
            1: Task(1, "Préparation du site", td, None, 5, 10, 7, None, "Équipe A", "Bulldozer", progress=30),
            2: Task(2, "Construction route d'accès", None, None, 7, 12, 9, None, "Équipe B", "Excavatrice", dependencies="1", progress=15),
            3: Task(3, "Forage initial", None, None, 10, 18, 14, None, "Équipe C", "Foreuse", dependencies="1", progress=50),
            4: Task(4, "Installation équipements", None, None, 3, 8, 5, None, "Équipe D", "Grue", dependencies="2,3", progress=0),
        }

    def scheduled_end(self) -> Optional[date]:
        """Calcule la date de fin basée sur start_date et duration_days, ou retourne end_date"""
        if self.start_date and self.duration_days:
            return self.start_date + timedelta(days=self.duration_days - 1)
        return self.end_date

    def get_expected_duration(self) -> Optional[float]:
        """Retourne la durée attendue (stochastique en priorité, sinon durée en jours)"""
        if self.duration_stochastic is not None:
            return self.duration_stochastic
        elif self.duration_days is not None:
            return float(self.duration_days)
        elif self.duration_optimistic is not None and self.duration_pessimistic is not None:
            # Calcul PERT simple si pas encore calculé
            probable = self.duration_probable or (self.duration_optimistic + self.duration_pessimistic) / 2
            return (self.duration_optimistic + 4 * probable + self.duration_pessimistic) / 6
        return None

    def get_actual_duration(self) -> Optional[int]:
        """Retourne la durée réelle si la tâche est terminée"""
        if self.start_date and self.end_date:
            return (self.end_date - self.start_date).days + 1
        return None

    def get_remaining_duration(self) -> Optional[float]:
        """Retourne la durée restante basée sur la progression"""
        expected = self.get_expected_duration()
        if expected and self.progress < 100:
            return expected * (100 - self.progress) / 100
        return 0 if self.progress == 100 else expected

    def get_stochastic_end_date(self) -> Optional[date]:
        """Retourne la date de fin basée sur la durée stochastique"""
        if self.start_date and self.duration_stochastic:
            return self.start_date + timedelta(days=int(round(self.duration_stochastic)))
        elif self.start_date and self.duration_days:
            return self.start_date + timedelta(days=self.duration_days - 1)
        return self.end_date

    def get_standard_deviation(self) -> Optional[float]:
        """Retourne l'écart-type de la tâche"""
        return self.standard_deviation

    def to_dict(self):
        d = asdict(self)
        d["start_date"] = self.start_date.isoformat() if self.start_date else None
        d["end_date"] = self.end_date.isoformat() if self.end_date else None
        d["projected_end_date"] = self.projected_end_date.isoformat() if self.projected_end_date else None
        return d

    def get_dependency_ids(self) -> List[int]:
        """Parse les dépendances et retourne une liste d'IDs"""
        if not self.dependencies:
            return []
        deps_str = self.dependencies.replace(";", ",")
        return [int(x.strip()) for x in deps_str.split(",") if x.strip().isdigit()]

    def is_complete_for_calculation(self) -> bool:
        """Vérifie si la tâche a assez d'informations pour les calculs"""
        has_dates = self.start_date and self.end_date
        has_start_duration = self.start_date and self.duration_days
        has_end_duration = self.end_date and self.duration_days
        return has_dates or has_start_duration or has_end_duration




    def calculate_stochastic_duration(self, task: Task) -> Optional[float]:
        """Calcule la durée stochastique selon la méthode PERT"""
        if task.duration_optimistic is None or task.duration_pessimistic is None:
            return task.duration_stochastic  # Garder la valeur existante si pas de données
            
        if task.duration_probable is None:
            task.duration_probable = (task.duration_optimistic + task.duration_pessimistic) / 2
        
        task.duration_stochastic = (task.duration_optimistic + 4 * task.duration_probable + task.duration_pessimistic) / 6
        return task.duration_stochastic

    def calculate_standard_deviation(self, task: Task) -> Optional[float]:
        """Calcule l'écart-type"""
        if task.duration_optimistic is None or task.duration_pessimistic is None:
            return task.standard_deviation  # Garder la valeur existante
            
        task.standard_deviation = (task.duration_pessimistic - task.duration_optimistic) / 6
        return task.standard_deviation

    def identify_critical_tasks(self, critical_path: List[int]):
        """Identifie les tâches critiques"""
        for task in self.tasks.values():
            task.is_critical = task.id in critical_path

    
    def calculate_buffer(self, task: Task, predecessor_std: Optional[float] = None) -> Optional[float]:
        """Calcule le buffer selon les règles définies"""
        if task.duration_optimistic is None or task.duration_pessimistic is None:
            return task.buffer  # Garder la valeur existante
            
        if task.standard_deviation is None:
            self.calculate_standard_deviation(task)
            
        if task.standard_deviation is None:
            return task.buffer
        
        # Déterminer le type de tâche
        num_predecessors = len(task.get_dependency_ids())
        is_multi_dependency = num_predecessors > 1
        
        if task.is_critical:
            # Tâche critique
            if predecessor_std is not None:
                combined_std = np.sqrt(predecessor_std**2 + task.standard_deviation**2)
            else:
                combined_std = task.standard_deviation
            task.buffer = self.coefficient_critical * combined_std
        else:
            # Tâche non-critique
            task.buffer = self.coefficient_non_critical * task.standard_deviation
        
        # Appliquer le multiplicateur pour les tâches multi-dépendances
        if is_multi_dependency and task.buffer is not None:
            task.buffer *= self.multiplier_multi_dependencies
            
        return task.buffer

    def calculate_projected_end_date(self, task: Task) -> Optional[date]:
        """Calcule la date de fin projetée """
        # Ne calculer que si on a les données nécessaires ET une progression > 0
        if not task.start_date or not task.duration_stochastic or task.progress == 0:
            return task.projected_end_date  # Garder la valeur existante
        
        # Nombre de jours de travail réalisés
        work_done_days = task.duration_stochastic * (task.progress / 100)
        
        # Nombre de jours écoulés depuis le début
        days_elapsed = (date.today() - task.start_date).days
        
        # Gérer le cas où la tâche vient de commencer ou dans le futur
        if days_elapsed <= 0:
            days_elapsed = 1  # Au minimum 1 jour pour éviter la division par zéro
        
        # Vitesse de réalisation
        execution_speed = work_done_days / days_elapsed
        
        # Durée réelle estimée
        if execution_speed > 0:
            real_duration = task.duration_stochastic / execution_speed
        else:
            # Si aucun travail n'a été fait mais qu'il y a une progression déclarée,
            # utiliser la durée stochastique de base
            real_duration = task.duration_stochastic
        
        # Date de fin projetée
        task.projected_end_date = task.start_date + timedelta(days=int(round(real_duration)))
        
        return task.projected_end_date

    def _calculate_start_from_dependencies(self, task: Task) -> Optional[date]:
        """Calcule la date de début basée sur les dépendances"""
        dep_ids = task.get_dependency_ids()
        if not dep_ids:
            return task.start_date  # Garder la valeur existante si pas de dépendances

        latest_end = None
        dep_type = task.dependency_type.upper()
        lag_days = task.lag

        for dep_id in dep_ids:
            parent = self.tasks.get(dep_id)
            if not parent:
                continue

            parent_date = None

            if dep_type == "FS":  # Finish to Start
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
            elif dep_type == "SF":  # Start to Finish
                parent_date = parent.start_date
                if parent_date and task.duration_days:
                    parent_date = parent_date - timedelta(days=task.duration_days - 1) + timedelta(days=lag_days)

            if parent_date and (latest_end is None or parent_date > latest_end):
                latest_end = parent_date

        return latest_end
        
##### Chemin Critique 
    def find_critical_path(self) -> List[int]:
        """Trouve le chemin critique"""
        if not self.tasks:
            return []

        # Étape 1: Calcul des dates au plus tôt (Forward Pass)
        earliest_start = {}
        earliest_finish = {}
        
        # Tri topologique pour traiter les tâches dans le bon ordre
        sorted_tasks = self._topological_sort()
        
        for task_id in sorted_tasks:
            task = self.tasks[task_id]
            
            # Date de début au plus tôt
            dep_ids = task.get_dependency_ids()
            if not dep_ids:
                # Tâche sans dépendance
                earliest_start[task_id] = task.start_date or date.today()
            else:
                # Tâche avec dépendances
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
            
            # Date de fin au plus tôt
            duration = task.duration_days or (int(task.duration_stochastic) if task.duration_stochastic else 1)
            earliest_finish[task_id] = earliest_start[task_id] + timedelta(days=duration - 1)

        # Étape 2: Calcul des dates au plus tard (Backward Pass)
        latest_start = {}
        latest_finish = {}
        
        # Trouver la date de fin du projet
        if earliest_finish:
            project_end = max(earliest_finish.values())
        else:
            return []
        
        # Traiter les tâches dans l'ordre inverse
        for task_id in reversed(sorted_tasks):
            task = self.tasks[task_id]
            
            # Trouver les successeurs
            successors = [t.id for t in self.tasks.values() if task_id in t.get_dependency_ids()]
            
            if not successors:
                # Tâche finale
                latest_finish[task_id] = project_end
            else:
                # Tâche avec successeurs
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
            
            # Date de début au plus tard
            duration = task.duration_days or (int(task.duration_stochastic) if task.duration_stochastic else 1)
            latest_start[task_id] = latest_finish[task_id] - timedelta(days=duration - 1)

        # Étape 3: Identifier les tâches critiques (marge = 0)
        critical_tasks = []
        for task_id in self.tasks.keys():
            if (task_id in earliest_start and task_id in latest_start and
                earliest_start[task_id] == latest_start[task_id]):
                critical_tasks.append(task_id)
        
        return critical_tasks

    def auto_calculate_all_tasks(self, max_iterations: int = 10):
        """Calcule automatiquement toutes les tâches manquantes"""
        changes_made = True
        iteration = 0

        # Première passe : calculer les durées stochastiques et écart-types SANS réinitialiser
        for task in self.tasks.values():
            if task.duration_optimistic is not None and task.duration_pessimistic is not None:
                # Ne recalculer que si pas déjà calculé
                if task.duration_stochastic is None:
                    self.calculate_stochastic_duration(task)
                if task.standard_deviation is None:
                    self.calculate_standard_deviation(task)
                if task.duration_days is None and task.duration_stochastic is not None:
                    task.duration_days = int(round(task.duration_stochastic))

        # Deuxième passe : calculer les dates avec gestion des dépendances
        while changes_made and iteration < max_iterations:
            changes_made = False
            iteration += 1

            # Trier les tâches par ordre topologique
            sorted_tasks = self._topological_sort()

            for task_id in sorted_tasks:
                task = self.tasks[task_id]
                
                # Sauvegarder l'état précédent pour détecter les changements
                old_start = task.start_date
                old_end = task.end_date
                old_duration = task.duration_days

                # Calculer la date de début basée sur les dépendances
                if task.get_dependency_ids() and not task.start_date:
                    calculated_start = self._calculate_start_from_dependencies(task)
                    if calculated_start and calculated_start != task.start_date:
                        task.start_date = calculated_start
                        changes_made = True

                # Calculer la date de fin si start et durée sont présents
                if task.start_date and task.duration_days and not task.end_date:
                    new_end = task.start_date + timedelta(days=task.duration_days - 1)
                    if new_end != task.end_date:
                        task.end_date = new_end
                        changes_made = True

                # Calculer la date de début si fin et durée sont présents
                elif task.end_date and task.duration_days and not task.start_date:
                    new_start = task.end_date - timedelta(days=task.duration_days - 1)
                    if new_start != task.start_date:
                        task.start_date = new_start
                        changes_made = True

                # Calculer la durée si les deux dates sont présentes
                elif task.start_date and task.end_date and not task.duration_days:
                    duration = (task.end_date - task.start_date).days + 1
                    if duration != task.duration_days:
                        task.duration_days = duration
                        changes_made = True

        # Troisième passe : identifier le chemin critique
        try:
            critical_path = self.find_critical_path()
            self.identify_critical_tasks(critical_path)
        except Exception as e:
            if hasattr(st, 'warning'):
                st.warning(f"Erreur lors du calcul du chemin critique: {e}")

        # Quatrième passe : calculer les buffers
        sorted_tasks = self._topological_sort()
        for task_id in sorted_tasks:
            task = self.tasks[task_id]
            
            # Ne calculer le buffer que si les données nécessaires sont présentes
            if (task.duration_optimistic is not None and 
                task.duration_pessimistic is not None and 
                task.buffer is None):  # Ne recalculer que si pas déjà calculé
                
                predecessor_std = None
                
                # Obtenir l'écart-type du prédécesseur pour les tâches critiques
                if task.is_critical and task.get_dependency_ids():
                    dep_id = task.get_dependency_ids()[0]
                    dep_task = self.tasks.get(dep_id)
                    if dep_task and dep_task.standard_deviation is not None:
                        predecessor_std = dep_task.standard_deviation
                
                self.calculate_buffer(task, predecessor_std)

        # Cinquième passe : calculer les dates de fin projetées - CORRECTION ICI
        for task in self.tasks.values():
            # Recalculer TOUJOURS la date de fin projetée si on a une progression > 0
            if task.progress > 0 and task.start_date and task.duration_stochastic:
                self.calculate_projected_end_date(task)

        # Sixième passe : ajuster les dates de fin avec buffers si nécessaire
        for task in self.tasks.values():
            if (task.start_date and task.duration_stochastic is not None and 
                task.buffer is not None and task.duration_days):
                # Calculer la durée totale avec buffer
                total_duration = int(round(task.duration_stochastic + task.buffer))
                if total_duration != task.duration_days:
                    task.duration_days = total_duration
                    # Recalculer la date de fin
                    if task.start_date:
                        task.end_date = task.start_date + timedelta(days=task.duration_days - 1)

    def _topological_sort(self) -> List[int]:
        """Tri topologique des tâches basé sur les dépendances"""
        result = []
        visited = set()
        temp_visited = set()

        def visit(task_id: int):
            if task_id in temp_visited:
                return  # Cycle détecté, ignorer
            if task_id in visited:
                return

            temp_visited.add(task_id)

            # Visiter les dépendances d'abord
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

    def validate_task_data(self) -> List[str]:
        """Valide les données des tâches et retourne les erreurs"""
        errors = []
        for task in self.tasks.values():
            if not task.name.strip():
                errors.append(f"Tâche {task.id}: Le nom est obligatoire")

            # Vérifier les dépendances circulaires
            if self._has_circular_dependency(task.id):
                errors.append(f"Tâche {task.id}: Dépendance circulaire détectée")

            # Vérifier que les dépendances existent
            for dep_id in task.get_dependency_ids():
                if dep_id not in self.tasks:
                    errors.append(f"Tâche {task.id}: Dépendance {dep_id} inexistante")

        return errors

    def _has_circular_dependency(self, task_id: int, visited: set = None) -> bool:
        """Vérifie s'il y a une dépendance circulaire"""
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


# Fonction utilitaire pour mettre à jour les tâches - VERSION CORRIGÉE
def update_tasks_from_editor(tm: TaskManager, edited_df: pd.DataFrame):
    """Met à jour les tâches basé sur les données de l'éditeur"""
    # Récupérer les IDs actuels
    new_ids = set()
    for _, row in edited_df.iterrows():
        if pd.notna(row.get("ID")):
            new_ids.add(int(row["ID"]))

    # Supprimer les tâches supprimées
    for tid in list(tm.tasks.keys()):
        if tid not in new_ids:
            del tm.tasks[tid]

    # Mettre à jour ou créer les tâches
    for _, row in edited_df.iterrows():
        # Vérifier si la ligne a un nom de tâche
        if pd.isna(row.get("Nom de la tâche*")) or not str(row.get("Nom de la tâche*", "")).strip():
            continue

        name = str(row.get("Nom de la tâche*", "")).strip()
        tid = int(row["ID"]) if pd.notna(row.get("ID")) else tm.next_id()

        # Parser les dates avec gestion d'erreurs améliorée
        def parse_date(date_str):
            if pd.isna(date_str) or not str(date_str).strip():
                return None
            try:
                date_str = str(date_str).strip()
                if len(date_str) == 10 and '-' in date_str:
                    return date.fromisoformat(date_str)
            except (ValueError, TypeError):
                pass
            return None

        start_date = parse_date(row.get("Date début"))
        end_date = parse_date(row.get("Date fin"))
        projected_end_date = parse_date(row.get("Date de fin projection"))

        # Parser les valeurs numériques avec vérification améliorée
        def safe_int(value):
            if pd.isna(value) or value == "":
                return None
            try:
                return int(float(value))  # Convertir en float d'abord au cas où
            except (ValueError, TypeError):
                return None
            
        def safe_float(value):
            if pd.isna(value) or value == "":
                return None
            try:
                return float(value)
            except (ValueError, TypeError):
                return None

        # Récupérer la tâche existante ou créer une nouvelle
        existing_task = tm.tasks.get(tid)
        
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
            duration_stochastic=safe_float(row.get("Durée stochastique")) if existing_task is None else (safe_float(row.get("Durée stochastique")) or existing_task.duration_stochastic),
            duration_days=safe_int(row.get("Durée (jours)")),
            dependencies=str(row.get("Dépendance", "")).strip(),
            dependency_type=str(row.get("Type Dép.", "FS")).strip(),
            lag=safe_int(row.get("Décalage (j)")) or 0,
            progress=safe_int(row.get("Progression")) or 0,
            comments=str(row.get("Commentaires", "")).strip() or None,
            standard_deviation=safe_float(row.get("Écart type")) if existing_task is None else (safe_float(row.get("Écart type")) or existing_task.standard_deviation),
            buffer=safe_float(row.get("Buffer")) if existing_task is None else (safe_float(row.get("Buffer")) or existing_task.buffer),
            projected_end_date=projected_end_date if existing_task is None else (projected_end_date or existing_task.projected_end_date),
            is_critical=bool(row.get("État critique")) if pd.notna(row.get("État critique")) else (existing_task.is_critical if existing_task else None),
        )

        tm.tasks[tid] = task


# ---------------------------- UI ----------------------------

st.set_page_config(page_title="⚒️ Gestionnaire de Tâches", layout="wide")


def main():
    # Sidebar pour navigation
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.selectbox(
        "Choisir une page",
        ["📊 Gestion des Tâches", "📈 Visualisations & Statistiques"]
    )

    # Initialiser le gestionnaire de tâches
    if "task_manager" not in st.session_state:
        st.session_state.task_manager = TaskManager()

    tm = st.session_state.task_manager

    # Navigation
    if page == "📊 Gestion des Tâches":
        show_task_management_page(tm)
    elif page == "📈 Visualisations & Statistiques":
        try:
            from visualizations import show_visualizations_page
            show_visualizations_page(tm)
        except Exception as e:
            st.error("❌ Erreur lors du chargement des visualisations :")
            st.code(str(e))
    else:
        st.warning("Page inconnue")


def show_task_management_page(tm: TaskManager):
    """Page principale de gestion des tâches - VERSION CORRIGÉE"""
    st.title("⚒️ Gestionnaire de Tâches Avancé")
    st.markdown("---")

    # Configuration des coefficients
    st.sidebar.header("⚙️ Configuration des coefficients")
    tm.coefficient_non_critical = st.sidebar.number_input(
        "Coefficient non-critique", 
        value=float(tm.coefficient_non_critical), 
        min_value=0.1, 
        max_value=2.0, 
        step=0.1
    )
    tm.coefficient_critical = st.sidebar.number_input(
        "Coefficient critique", 
        value=float(tm.coefficient_critical), 
        min_value=0.1, 
        max_value=2.0, 
        step=0.1
    )
    tm.multiplier_multi_dependencies = st.sidebar.number_input(
        "Multiplicateur multi-dépendances", 
        value=float(tm.multiplier_multi_dependencies), 
        min_value=1.0, 
        max_value=3.0, 
        step=0.1
    )

    # Actions dans la sidebar
    st.sidebar.header("🛠️ Actions")

    if st.sidebar.button("🔄 Réinitialiser avec exemples", help="Charge des tâches d'exemple"):
        tm.create_sample()
        tm.save()
        st.rerun()

    if st.sidebar.button("💾 Sauvegarder", help="Sauvegarde manuelle"):
        tm.save()
        st.sidebar.success("✅ Sauvegardé!")

    # Validation des données
    errors = tm.validate_task_data()
    if errors:
        st.sidebar.error("⚠️ Erreurs détectées:")
        for error in errors:
            st.sidebar.write(f"• {error}")

    # Interface principale
    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("📊 Tableau de Gestion des Tâches")

    with col2:
        st.write(f"**Total:** {len(tm.tasks)} tâches")

    # Créer le dataframe pour l'éditeur
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

    # Si aucune tâche, ajouter une ligne vide
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

    # Configuration des colonnes
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

    # Éditeur de données
    edited_df = st.data_editor(
        pd.DataFrame(records),
        use_container_width=True,
        num_rows="dynamic",
        column_config=column_config,
        key="task_editor"
    )

    # Boutons d'action
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("🧮 **Calculer les champs manquants**", type="primary", help="Calcule automatiquement les dates et durées manquantes"):
            with st.spinner("Calcul en cours..."):
                try:
                    # Mettre à jour les tâches depuis l'éditeur
                    update_tasks_from_editor(tm, edited_df)
                    
                    # Effectuer les calculs
                    tm.auto_calculate_all_tasks()
                    
                    # Sauvegarder
                    tm.save()
                    
                    # Mettre à jour le session state
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

    # Résumé rapide
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

    # Aide
    with st.expander("ℹ️ Aide et Instructions"):
        st.markdown("""
        ### 📝 Comment utiliser ce gestionnaire de tâches :

        **Champs obligatoires :**
        - **Nom de la tâche** : Obligatoire pour chaque tâche
        - **Durée optimiste** et **Durée pessimiste** : Requis pour le calcul automatique

        **Calculs automatiques :**
        - **Durée stochastique** : Calculée selon la formule PERT
        - **Écart type** : Calculé automatiquement
        - **Buffer** : Calculé selon les règles de gestion des risques
        - **Dates** : Calculées selon les dépendances et durées
        - **État critique** : Déterminé par l'algorithme du chemin critique
        - **Date de fin projection** : Calculée si progression > 0

        **Format des dépendances :**
        - Séparez les IDs par des virgules : `1,2,3`
        - Ou par des points-virgules : `1;2;3`

        **Actions disponibles :**
        - **Calculer les champs manquants** : Lance tous les calculs automatiques
        - **Appliquer les modifications** : Sauvegarde vos changements sans recalculer
        - Ajoutez/supprimez des lignes directement dans le tableau

        **⚡ Fonctionnalités de projection :**
        - La date de fin projetée se calcule automatiquement si la progression > 0
        - Le calcul prend en compte la vitesse d'exécution réelle
        """)


if __name__ == "__main__":
    main()
