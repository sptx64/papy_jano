import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from datetime import date, timedelta

def show_visualizations_page(tm):
    """Page principale des visualisations avec Gantt avancé"""
    st.title("📈 Visualisations & Statistiques")
    
    # --- Contrôles dans la sidebar ---
    st.sidebar.header("⚙️ Paramètres du Gantt")
    
    # Sélection du mode d'affichage
    view_mode = st.sidebar.selectbox(
        "📊 Regrouper par",
        ["Par tâches", "Par responsable", "Par équipement"]
    )
    
    # Contrôle de zoom horizontal
    zoom_option = st.sidebar.radio(
        "🔍 Zoom temporel",
        ["Tout voir", "1 mois", "3 mois", "6 mois"],
        horizontal=True
    )
    
    # Options d'affichage
    show_dependencies = st.sidebar.checkbox("Afficher les dépendances", value=True)
    show_critical_path = st.sidebar.checkbox("Afficher le chemin critique", value=True)
    show_uncertainty = st.sidebar.checkbox("Afficher l'incertitude", value=True)
    show_trend = st.sidebar.checkbox("Afficher les tendances", value=True)
    
    # Hauteur des barres de tendance
    trend_height = st.sidebar.slider("Hauteur des tendances", 0.1, 0.5, 0.3)
    
    today = date.today()
    critical_path = calculate_critical_path_simplified(tm)
    
    # --- MÉTRIQUES DE PERFORMANCE ---
    st.markdown("---")
    st.markdown("#### 📈 Métriques de Performance")
    
    # Calculer les métriques globales
    total_tasks = len([t for t in tm.tasks.values() if t.start_date])
    completed_tasks = len([t for t in tm.tasks.values() if t.progress == 100])
    delayed_tasks = 0
    accelerated_tasks = 0
    
    for task in tm.tasks.values():
        if task.start_date and task.progress > 0 and task.progress < 100:
            expected_duration = task.get_expected_duration() or 7
            expected_progress = min(100, max(0, (today - task.start_date).days / expected_duration * 100))
            if task.progress < expected_progress - 5:
                delayed_tasks += 1
            elif task.progress > expected_progress + 5:
                accelerated_tasks += 1
    
    # Affichage des métriques en colonnes
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        completion_rate = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
        st.metric(
            label="🎯 Taux de completion",
            value=f"{completion_rate:.1f}%",
            delta=f"{completed_tasks}/{total_tasks} tâches"
        )
    
    with metric_col2:
        on_time_rate = ((total_tasks - delayed_tasks) / total_tasks * 100) if total_tasks > 0 else 0
        st.metric(
            label="⏰ Tâches dans les temps",
            value=f"{on_time_rate:.1f}%",
            delta=f"-{delayed_tasks} en retard" if delayed_tasks > 0 else "✓ Aucun retard"
        )
    
    with metric_col3:
        st.metric(
            label="🚀 Tâches accélérées",
            value=f"{accelerated_tasks}",
            delta=f"+{(accelerated_tasks/total_tasks*100):.1f}%" if total_tasks > 0 else "0%"
        )
    
    with metric_col4:
        critical_tasks = len(critical_path)
        st.metric(
            label="⚠️ Chemin critique",
            value=f"{critical_tasks}",
            delta=f"{(critical_tasks/total_tasks*100):.1f}% du projet" if total_tasks > 0 else "0%"
        )
    
    # --- Onglets ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Diagramme de Gantt", 
        "📈 Statistiques", 
        "🔍 Analyse", 
        "📋 Rapport"
    ])
    
    with tab1:
        show_gantt_chart(tm, view_mode, zoom_option, show_dependencies, 
                        show_critical_path, show_uncertainty, show_trend, trend_height)
    
    with tab2:
        show_statistics(tm)
    
    with tab3:
        show_analysis(tm)
    
    with tab4:
        show_report(tm)

def show_statistics(tm):
    """Affiche les statistiques du projet"""
    st.subheader("📊 Statistiques du Projet")
    
    if not tm.tasks:
        st.warning("Aucune donnée à analyser")
        return
    
    # Statistiques générales
    total_tasks = len(tm.tasks)
    completed_tasks = len([t for t in tm.tasks.values() if t.progress == 100])
    in_progress_tasks = len([t for t in tm.tasks.values() if 0 < t.progress < 100])
    not_started_tasks = len([t for t in tm.tasks.values() if t.progress == 0])
    
    # Graphique en secteurs pour la répartition des tâches
    fig_pie = go.Figure(data=[go.Pie(
        labels=['Terminées', 'En cours', 'Non commencées'],
        values=[completed_tasks, in_progress_tasks, not_started_tasks],
        hole=0.4,
        marker_colors=['#4CAF50', '#2196F3', '#FFC107']
    )])
    
    fig_pie.update_layout(
        title="Répartition des Tâches par Statut",
        height=400
    )
    
    st.plotly_chart(fig_pie, use_container_width=True)
    
    # Statistiques détaillées par responsable
    st.markdown("#### 👥 Répartition par Responsable")
    
    responsable_stats = {}
    for task in tm.tasks.values():
        resp = task.responsible or "Non assigné"
        if resp not in responsable_stats:
            responsable_stats[resp] = {'total': 0, 'completed': 0, 'progress_sum': 0}
        
        responsable_stats[resp]['total'] += 1
        responsable_stats[resp]['progress_sum'] += task.progress
        if task.progress == 100:
            responsable_stats[resp]['completed'] += 1
    
    # Tableau des statistiques par responsable
    resp_data = []
    for resp, stats in responsable_stats.items():
        avg_progress = stats['progress_sum'] / stats['total'] if stats['total'] > 0 else 0
        completion_rate = stats['completed'] / stats['total'] * 100 if stats['total'] > 0 else 0
        
        resp_data.append({
            'Responsable': resp,
            'Tâches totales': stats['total'],
            'Tâches terminées': stats['completed'],
            'Taux de completion': f"{completion_rate:.1f}%",
            'Progression moyenne': f"{avg_progress:.1f}%"
        })
    
    df_resp = pd.DataFrame(resp_data)
    st.dataframe(df_resp, use_container_width=True)

def show_analysis(tm):
    """Affiche l'analyse détaillée du projet"""
    st.subheader("🔍 Analyse du Projet")
    
    if not tm.tasks:
        st.warning("Aucune donnée à analyser")
        return
    
    today = date.today()
    
    # Analyse des risques
    st.markdown("#### ⚠️ Analyse des Risques")
    
    risk_tasks = []
    for task in tm.tasks.values():
        if task.start_date and task.progress < 100:
            # Calculer le risque de retard
            expected_duration = task.get_expected_duration() or 7
            expected_progress = min(100, max(0, (today - task.start_date).days / expected_duration * 100))
            delay_risk = max(0, expected_progress - task.progress)
            
            if delay_risk > 10:  # Risque significatif
                risk_level = "Rouge" if delay_risk > 30 else "Orange"
                risk_tasks.append({
                    'Tâche': task.name,
                    'Responsable': task.responsible or "Non assigné",
                    'Progression actuelle': f"{task.progress}%",
                    'Progression attendue': f"{expected_progress:.1f}%",
                    'Écart': f"{delay_risk:.1f}%",
                    'Niveau de risque': risk_level
                })
    
    if risk_tasks:
        df_risks = pd.DataFrame(risk_tasks)
        st.dataframe(df_risks, use_container_width=True)
    else:
        st.success("✅ Aucun risque de retard significatif détecté")
    
    # Analyse des dépendances critiques
    st.markdown("#### 🔗 Dépendances Critiques")
    
    critical_path = calculate_critical_path_simplified(tm)
    if critical_path:
        st.warning(f"⚠️ Le chemin critique contient {len(critical_path)} tâches")
        
        critical_tasks_info = []
        for task_id in critical_path:
            task = tm.tasks.get(task_id)
            if task:
                critical_tasks_info.append({
                    'ID': task.id,
                    'Tâche': task.name,
                    'Responsable': task.responsible or "Non assigné",
                    'Progression': f"{task.progress}%",
                    'Durée estimée': f"{task.get_expected_duration() or 0} jours"
                })
        
        if critical_tasks_info:
            df_critical = pd.DataFrame(critical_tasks_info)
            st.dataframe(df_critical, use_container_width=True)
    else:
        st.info("ℹ️ Aucun chemin critique identifié")
    
    # Recommandations
    st.markdown("#### 💡 Recommandations")
    
    recommendations = []
    
    # Analyser les tâches en retard
    delayed_count = 0
    for task in tm.tasks.values():
        if task.start_date and task.progress < 100:
            expected_duration = task.get_expected_duration() or 7
            expected_progress = min(100, max(0, (today - task.start_date).days / expected_duration * 100))
            if task.progress < expected_progress - 5:
                delayed_count += 1
    
    if delayed_count > 0:
        recommendations.append(f"🔴 Action urgente: {delayed_count} tâche(s) en retard nécessitent une attention immédiate")
    
    # Analyser la charge de travail
    workload = {}
    for task in tm.tasks.values():
        if task.responsible and task.progress < 100:
            workload[task.responsible] = workload.get(task.responsible, 0) + 1
    
    if workload:
        max_workload = max(workload.values())
        overloaded = [resp for resp, count in workload.items() if count > max_workload * 0.8]
        if overloaded:
            recommendations.append(f"⚖️ Redistribuer la charge de {', '.join(overloaded)}")
    
    # Analyser les dépendances
    if len(critical_path) > len(tm.tasks) * 0.3:
        recommendations.append("🔗 Trop de tâches critiques, envisager la parallélisation")
    
    if not recommendations:
        recommendations.append("✅ Projet sain: Aucune action corrective majeure requise")
    
    for rec in recommendations:
        st.markdown(f"- {rec}")

def show_report(tm):
    """Génère un rapport complet du projet - VERSION CORRIGÉE"""
    st.subheader("📋 Rapport de Projet")
    
    if not tm.tasks:
        st.warning("Aucune donnée pour générer le rapport")
        return
    
    today = date.today()
    
    # En-tête du rapport
    st.markdown("### 📊 Rapport de Suivi de Projet")
    st.markdown(f"**Date du rapport**: {today.strftime('%d/%m/%Y')}")
    st.markdown(f"**Nombre total de tâches**: {len(tm.tasks)}")
    
    # Calculs pour le résumé exécutif
    completed_tasks = len([t for t in tm.tasks.values() if t.progress == 100])
    total_progress = sum(t.progress for t in tm.tasks.values()) / len(tm.tasks) if tm.tasks else 0
    
    # Calculer les tâches en retard et à l'heure
    tasks_with_dates = [t for t in tm.tasks.values() if t.start_date]
    on_time_tasks = 0
    delayed_tasks_count = 0
    ahead_tasks = 0
    not_started_with_schedule = 0
    
    delayed_tasks_details = []
    
    for task in tasks_with_dates:
        expected_duration = task.get_expected_duration() or 7
        
        if task.progress == 0:
            # Tâche pas encore commencée
            if task.start_date <= today:
                not_started_with_schedule += 1
                # Considérer comme en retard si devrait avoir commencé
                delayed_tasks_details.append({
                    'Tâche': task.name,
                    'Type de retard': 'Non commencée',
                    'Retard (jours)': (today - task.start_date).days,
                    'Progression actuelle': f"{task.progress}%",
                    'Progression attendue': f"{min(100, max(0, (today - task.start_date).days / expected_duration * 100)):.1f}%"
                })
                delayed_tasks_count += 1
        elif task.progress < 100:
            # Tâche en cours
            expected_progress = min(100, max(0, (today - task.start_date).days / expected_duration * 100))
            
            if task.progress < expected_progress - 10:  # Seuil de retard significatif
                delay_days = int((expected_progress - task.progress) / 100 * expected_duration)
                delayed_tasks_details.append({
                    'Tâche': task.name,
                    'Type de retard': 'En cours avec retard',
                    'Retard (jours)': delay_days,
                    'Progression actuelle': f"{task.progress}%",
                    'Progression attendue': f"{expected_progress:.1f}%"
                })
                delayed_tasks_count += 1
            elif task.progress > expected_progress + 15:  # En avance significative
                ahead_tasks += 1
                on_time_tasks += 1
            else:
                on_time_tasks += 1
        else:
            # Tâche terminée
            on_time_tasks += 1
    
    # Déterminer le statut basé sur les retards réels
    total_tasks_with_schedule = len(tasks_with_dates)
    if total_tasks_with_schedule > 0:
        delay_percentage = (delayed_tasks_count / total_tasks_with_schedule) * 100
        
        if delay_percentage == 0:
            status = "🟢 Excellent - Aucun retard"
        elif delay_percentage <= 10:
            status = "🟢 Dans les temps"
        elif delay_percentage <= 25:
            status = "🟠 Attention requise"
        elif delay_percentage <= 50:
            status = "🟡 Retards modérés"
        else:
            status = "🔴 Retard significatif"
    else:
        status = "⚪ Planning non défini"
    
    # Résumé exécutif amélioré
    st.markdown("#### 📈 Résumé Exécutif")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📊 Avancement global",
            value=f"{total_progress:.1f}%",
            delta=f"{completed_tasks}/{len(tm.tasks)} terminées"
        )
    
    with col2:
        st.metric(
            label="⏰ Tâches à l'heure",
            value=f"{on_time_tasks}",
            delta=f"{(on_time_tasks/total_tasks_with_schedule*100):.1f}%" if total_tasks_with_schedule > 0 else "N/A"
        )
    
    with col3:
        st.metric(
            label="⚠️ Tâches en retard",
            value=f"{delayed_tasks_count}",
            delta=f"-{(delayed_tasks_count/total_tasks_with_schedule*100):.1f}%" if total_tasks_with_schedule > 0 else "N/A",
            delta_color="inverse"
        )
    
    with col4:
        st.metric(
            label="🚀 Tâches en avance",
            value=f"{ahead_tasks}",
            delta=f"+{(ahead_tasks/total_tasks_with_schedule*100):.1f}%" if total_tasks_with_schedule > 0 else "N/A"
        )
    
    # Calculer le taux de retard
    delay_rate = (delayed_tasks_count/total_tasks_with_schedule*100) if total_tasks_with_schedule > 0 else 0
    
    # Afficher le statut général
    st.markdown(f"**📊 Statut général du projet**: {status}")
    st.markdown("**📋 Analyse détaillée**:")
    st.markdown(f"- Tâches avec planning défini: {total_tasks_with_schedule}/{len(tm.tasks)}")
    st.markdown(f"- Taux de retard: {delay_rate:.1f}%")
    st.markdown(f"- Performance globale: {total_progress:.1f}% d'avancement")
    
    # Analyse des retards avec plus de détails
    if delayed_tasks_details:
        st.markdown("#### ⚠️ Analyse des Retards")
        
        # Séparer par type de retard
        not_started = [t for t in delayed_tasks_details if t['Type de retard'] == 'Non commencée']
        in_progress_delayed = [t for t in delayed_tasks_details if t['Type de retard'] == 'En cours avec retard']
        
        if not_started:
            st.markdown("**🔴 Tâches qui auraient dû commencer:**")
            df_not_started = pd.DataFrame(not_started)
            st.dataframe(df_not_started.drop('Type de retard', axis=1), use_container_width=True)
        
        if in_progress_delayed:
            st.markdown("**🟠 Tâches en cours avec retard:**")
            df_delayed = pd.DataFrame(in_progress_delayed)
            st.dataframe(df_delayed.drop('Type de retard', axis=1), use_container_width=True)
        
        # Recommandations basées sur les retards
        st.markdown("#### 💡 Recommandations d'Actions")
        
        if len(not_started) > 0:
            st.error(f"🔴 Action urgente: {len(not_started)} tâche(s) en retard de démarrage - Revoir les ressources et priorités")
        
        if len(in_progress_delayed) > 0:
            st.warning(f"🟠 Attention: {len(in_progress_delayed)} tâche(s) en cours de réalisation avec retard - Suivre de près")
        
        # Calculer l'impact sur la fin de projet
        max_delay = max([t['Retard (jours)'] for t in delayed_tasks_details])
        if max_delay > 0:
            st.info(f"📅 Impact estimé: Retard maximum de {max_delay} jours pouvant affecter la livraison finale")
    
    else:
        st.success("#### ✅ Aucun retard significatif détecté")
        st.markdown("🎉 Félicitations ! Votre projet se déroule selon le planning prévu.")
        st.markdown("Points positifs identifiés:")
        st.markdown("- Toutes les tâches respectent leur calendrier")
        st.markdown("- Aucun retard de démarrage")
        st.markdown("- Progression conforme aux attentes")
        
        if ahead_tasks > 0:
            st.info(f"🚀 Bonus: {ahead_tasks} tâche(s) en avance sur le planning !")
    
    # Détail par responsable
    st.markdown("#### 👥 Performance par Responsable")
    
    responsable_data = []
    for resp in set(t.responsible for t in tm.tasks.values() if t.responsible):
        resp_tasks = [t for t in tm.tasks.values() if t.responsible == resp and t.start_date]
        if not resp_tasks:
            continue
            
        resp_progress = sum(t.progress for t in resp_tasks) / len(resp_tasks)
        resp_completed = len([t for t in resp_tasks if t.progress == 100])
        total_resp_tasks = len(resp_tasks)
        
        # Calculer les retards pour ce responsable
        resp_delayed = 0
        resp_on_time = 0
        
        for task in resp_tasks:
            if task.progress < 100:
                expected_duration = task.get_expected_duration() or 7
                expected_progress = min(100, max(0, (today - task.start_date).days / expected_duration * 100))
                
                if task.progress == 0 and task.start_date <= today:
                    resp_delayed += 1
                elif task.progress < expected_progress - 10:
                    resp_delayed += 1
                else:
                    resp_on_time += 1
            else:
                resp_on_time += 1
        
        # Déterminer le statut du responsable
        if resp_delayed == 0:
            resp_status = "🟢 A l'heure"
        elif resp_delayed / total_resp_tasks <= 0.25:
            resp_status = "🟠 Attention"
        else:
            resp_status = "🔴 Retards"
        
        responsable_data.append({
            'Responsable': resp,
            'Tâches assignées': total_resp_tasks,
            'Tâches terminées': resp_completed,
            'Tâches en retard': resp_delayed,
            'Tâches à l\'heure': resp_on_time,
            'Progression moyenne': f"{resp_progress:.1f}%",
            'Statut': resp_status
        })
    
    if responsable_data:
        df_resp = pd.DataFrame(responsable_data)
        st.dataframe(df_resp, use_container_width=True)
    
    # Prochaines échéances
    st.markdown("#### 📅 Prochaines Échéances (14 prochains jours)")
    
    upcoming_tasks = []
    next_two_weeks = today + timedelta(days=14)
    
    for task in tm.tasks.values():
        end_date = task.get_stochastic_end_date()
        if end_date and today <= end_date <= next_two_weeks and task.progress < 100:
            days_remaining = (end_date - today).days
            
            # Déterminer l'urgence
            if days_remaining <= 2:
                urgency = "🔴 Urgent"
            elif days_remaining <= 5:
                urgency = "🟠 Proche"
            else:
                urgency = "🟢 Normal"
            
            upcoming_tasks.append({
                'Tâche': task.name,
                'Responsable': task.responsible or "Non assigné",
                'Date de fin prévue': end_date.strftime('%d/%m/%Y'),
                'Jours restants': days_remaining,
                'Progression': f"{task.progress}%",
                'Urgence': urgency
            })
    
    if upcoming_tasks:
        df_upcoming = pd.DataFrame(upcoming_tasks)
        df_upcoming = df_upcoming.sort_values('Jours restants')
        st.dataframe(df_upcoming, use_container_width=True)
    else:
        st.info("ℹ️ Aucune échéance dans les 14 prochains jours")
    
    # Indicateurs de santé du projet
    st.markdown("#### 🏥 Indicateurs de Santé du Projet")
    
    health_col1, health_col2, health_col3 = st.columns(3)
    
    with health_col1:
        # Indicateur de respect des délais
        if delayed_tasks_count == 0:
            st.success("🎯 Délais: Parfait")
        elif delayed_tasks_count <= total_tasks_with_schedule * 0.1:
            st.info("🎯 Délais: Bon")
        elif delayed_tasks_count <= total_tasks_with_schedule * 0.3:
            st.warning("🎯 Délais: Attention")
        else:
            st.error("🎯 Délais: Critique")
    
    with health_col2:
        # Indicateur de progression
        if total_progress >= 75:
            st.success("📈 Progression: Excellente")
        elif total_progress >= 50:
            st.info("📈 Progression: Bonne")
        elif total_progress >= 25:
            st.warning("📈 Progression: Modérée")
        else:
            st.error("📈 Progression: Faible")
    
    with health_col3:
        # Indicateur de dynamique
        if ahead_tasks > delayed_tasks_count:
            st.success("🚀 Dynamique: Positive")
        elif ahead_tasks == delayed_tasks_count:
            st.info("⚖️ Dynamique: Équilibrée")
        else:
            st.warning("📊 Dynamique: À améliorer")
    
    # Bouton d'export (simulation)
    st.markdown("---")
    if st.button("📥 Exporter le Rapport Détaillé (PDF)", type="primary"):
        st.success("🎉 Rapport exporté avec succès ! (Fonctionnalité simulée)")
        st.balloons()

def calculate_date_range(zoom_option: str) -> tuple:
    """Calcule la plage de dates selon l'option de zoom"""
    today = date.today()
    
    if zoom_option == "1 mois":
        start = today - timedelta(days=15)
        end = today + timedelta(days=15)
    elif zoom_option == "3 mois":
        start = today - timedelta(days=45)
        end = today + timedelta(days=45)
    elif zoom_option == "6 mois":
        start = today - timedelta(days=90)
        end = today + timedelta(days=90)
    else:  # "Tout voir"
        return None, None
    
    return start, end

def calculate_critical_path_simplified(tm) -> set:
    """
    Version simplifiée pour les besoins pratiques
    Calcule le chemin critique du projet en tenant compte des durées
    """
    if not tm.tasks:
        return set()
    
    # Trouver les tâches terminales (qui n'ont pas de successeurs)
    all_dependencies = set()
    for task in tm.tasks.values():
        all_dependencies.update(task.get_dependency_ids())
    
    terminal_tasks = [tid for tid in tm.tasks.keys() if tid not in all_dependencies]
    
    # Pour chaque tâche terminale, calculer le chemin le plus long
    max_duration = 0
    critical_path = []
    
    for terminal_id in terminal_tasks:
        # Remonter le chemin
        path, path_duration = find_longest_path_to_root(tm, terminal_id)
        if path_duration > max_duration:
            max_duration = path_duration
            critical_path = path
    
    return set(critical_path)

def find_longest_path_to_root(tm, task_id, visited=None):
    """
    Trouve le chemin le plus long menant à une tâche racine
    """
    if visited is None:
        visited = set()
    
    if task_id in visited:
        return [], 0  # Cycle détecté
    
    visited.add(task_id)
    task = tm.tasks.get(task_id)
    if not task:
        return [], 0
    
    deps = task.get_dependency_ids()
    
    # Si c'est une tâche racine
    if not deps:
        return [task_id], task.get_expected_duration() or 1
    
    # Trouver le chemin le plus long parmi les dépendances
    longest_path = []
    max_duration = 0
    task_duration = task.get_expected_duration() or 1
    
    for dep_id in deps:
        path, duration = find_longest_path_to_root(tm, dep_id, visited.copy())
        if duration > max_duration:
            longest_path = path
            max_duration = duration
    
    return longest_path + [task_id], max_duration + task_duration

def calculate_trend_projection(task, today: date) -> tuple:
    """Calcule la tendance de performance et projection optimiste/pessimiste"""
    if not task.start_date or task.progress == 0:
        return None, None, 'stable'
    
    duration = task.get_expected_duration() or 7
    if duration <= 0:
        return None, None, 'stable'
    
    days_elapsed = max(1, (today - task.start_date).days)
    expected_progress = min(100, (days_elapsed / duration) * 100)
    
    # Calculer la tendance
    performance_ratio = task.progress / expected_progress if expected_progress > 0 else 1
    
    if performance_ratio > 1.1:
        trend = 'accelerating'
    elif performance_ratio < 0.9:
        trend = 'decelerating' 
    else:
        trend = 'stable'
    
    # Projections optimiste et pessimiste
    daily_rate = task.progress / days_elapsed
    remaining_progress = 100 - task.progress
    
    # Projection optimiste (rythme actuel + 20%)
    optimistic_rate = daily_rate * 1.2
    optimistic_days = int(remaining_progress / optimistic_rate) if optimistic_rate > 0 else float('inf')
    optimistic_end = today + timedelta(days=optimistic_days)
    
    # Projection pessimiste (rythme actuel - 20%)
    pessimistic_rate = daily_rate * 0.8
    pessimistic_days = int(remaining_progress / pessimistic_rate) if pessimistic_rate > 0 else float('inf')
    pessimistic_end = today + timedelta(days=pessimistic_days)
    
    return optimistic_end, pessimistic_end, trend

def calculate_cumulative_uncertainty(tm, task_id, visited=None) -> int:
    """Calcule l'incertitude cumulée pour une tâche et ses dépendances"""
    if visited is None:
        visited = set()
    
    if task_id in visited:
        return 0  # Éviter les cycles
    
    visited.add(task_id)
    task = tm.tasks.get(task_id)
    if not task:
        return 0
    
    # Incertitude de la tâche actuelle
    task_uncertainty = int(task.get_standard_deviation() or 0)
    
    # Incertitude cumulée des dépendances
    cumulative_uncertainty = task_uncertainty
    for dep_id in task.get_dependency_ids():
        dep_uncertainty = calculate_cumulative_uncertainty(tm, dep_id, visited.copy())
        cumulative_uncertainty += dep_uncertainty
    
    return cumulative_uncertainty

def show_gantt_chart(tm, view_mode: str, zoom_option: str, 
                    show_dependencies: bool, show_critical_path: bool,
                    show_uncertainty: bool, show_trend: bool, trend_height: float):
    """Affiche le diagramme de Gantt avec toutes les fonctionnalités"""
    
    if not tm.tasks:
        st.warning("Aucune tâche à afficher")
        return
    
    # Calculer les dates de zoom
    zoom_start, zoom_end = calculate_date_range(zoom_option)
    
    # Calculer le chemin critique
    critical_path = calculate_critical_path_simplified(tm) if show_critical_path else set()
    
    # Préparer les données selon le mode d'affichage
    fig = go.Figure()
    today = date.today()
    
    # Grouper les tâches selon le mode
    if view_mode == "Par responsable":
        grouped_tasks = {}
        for task in tm.tasks.values():
            key = task.responsible or "Non assigné"
            if key not in grouped_tasks:
                grouped_tasks[key] = []
            grouped_tasks[key].append(task)
    elif view_mode == "Par équipement":
        grouped_tasks = {}
        for task in tm.tasks.values():
            key = task.equipment or "Aucun équipement"
            if key not in grouped_tasks:
                grouped_tasks[key] = []
            grouped_tasks[key].append(task)
    else:  # Par tâches
        grouped_tasks = {"Toutes les tâches": list(tm.tasks.values())}
    
    # Créer les barres du Gantt
    y_pos = 0
    y_labels = []
    task_positions = {}
    
    # Palette de couleurs simplifiée
    colors = {
        'task_background': '#E8EAF6',        # Gris très clair pour le fond
        'task_progress': '#4CAF50',          # Vert pour la progression
        'delay': '#F44336',                  # Rouge pour le retard
        'trend_accelerating': '#8BC34A',     # Vert clair pour tendance positive
        'trend_decelerating': '#FFC107',     # Jaune pour tendance négative
        'critical_path': '#FFD700',          # Or pour chemin critique
        'dependencies': '#78909C',           # Gris pour dépendances
        'today_line': '#E91E63',             # Rose pour ligne du jour
        'uncertainty': '#9C27B0'             # Violet pour incertitude
    }
    
    # Créer une liste de toutes les tâches pour le traitement
    all_tasks = []
    for group_tasks in grouped_tasks.values():
        all_tasks.extend(group_tasks)
    
    # Trier les tâches par ID
    all_tasks.sort(key=lambda x: x.id)
    
    # Calculer la hauteur adaptative
    base_height_per_task = 35 if show_trend else 25
    chart_height = max(600, min(1400, len(all_tasks) * base_height_per_task))
    
    # Parcourir les groupes et les tâches
    group_separators = []
    current_y_pos = 0
    
    for group_name, tasks in grouped_tasks.items():
        if view_mode != "Par tâches":
            # Ajouter une ligne de groupe
            y_labels.append(f"<b>{group_name}</b>")
            task_positions[f"group_{group_name}"] = current_y_pos
            current_y_pos += 1
            
            if current_y_pos > 1:
                group_separators.append(current_y_pos - 0.5)
        
        # Trier les tâches par ID dans le groupe
        sorted_tasks = sorted(tasks, key=lambda x: x.id)
        
        for task in sorted_tasks:
            if not task.start_date:
                continue
            
            # Vérifier si dans la plage de zoom
            if zoom_start and zoom_end:
                task_end = task.get_stochastic_end_date()
                if task_end and task_end < zoom_start:
                    continue
                if task.start_date > zoom_end:
                    continue
            
            task_positions[task.id] = current_y_pos
            # Titres en noir pour meilleure visibilité
            y_labels.append(f"<span style='color:black'>{task.id}. {task.name}</span>")
            
            # Calculer les dates
            end_date = task.get_stochastic_end_date() or task.start_date
            
            # --- NIVEAU 1: PLANIFICATION NORMALE (Y = current_y_pos) ---
            
            # Hover text de base
            hover_text = (
                f"<b>{task.name}</b><br>" +
                f"Responsable: {task.responsible or 'Non assigné'}<br>" +
                f"Équipement: {task.equipment or 'Aucun'}<br>" +
                f"Début: {task.start_date.strftime('%Y-%m-%d')}<br>" +
                f"Fin prévue: {end_date.strftime('%Y-%m-%d')}<br>" +
                f"Progression: {task.progress}%"
            )
            
            # 1.1. Fond de la tâche (planification normale)
            fig.add_trace(go.Scatter(
                x=[task.start_date, end_date],
                y=[current_y_pos, current_y_pos],
                mode='lines',
                line=dict(color=colors['task_background'], width=22),
                name=f"Plan: {task.name}",
                showlegend=False,
                opacity=0.9,
                hovertemplate=hover_text + "<extra></extra>"
            ))
            
            # 1.2. Progression de la tâche (en VERT comme demandé)
            if task.progress > 0:
                expected_duration = task.get_expected_duration() or 7
                if expected_duration > 0:
                    progress_days = int(expected_duration * task.progress / 100)
                    progress_end = task.start_date + timedelta(days=progress_days - 1)
                    
                    fig.add_trace(go.Scatter(
                        x=[task.start_date, progress_end],
                        y=[current_y_pos, current_y_pos],
                        mode='lines',
                        line=dict(color=colors['task_progress'], width=22),
                        name=f"Progression: {task.name}",
                        showlegend=False,
                        hovertemplate=hover_text + "<extra></extra>"
                    ))
                    
                    # 1.3. Indicateur de retard
                    expected_progress = min(100, max(0, (today - task.start_date).days / expected_duration * 100))
                    is_late = task.progress < expected_progress and today > task.start_date and task.progress < 100
                    
                    if is_late and progress_end < end_date:
                        expected_days = int(expected_duration * expected_progress / 100)
                        expected_end = task.start_date + timedelta(days=expected_days - 1)
                        late_end = min(expected_end, end_date)
                        
                        if progress_end < late_end:
                            fig.add_trace(go.Scatter(
                                x=[progress_end + timedelta(days=1), late_end],
                                y=[current_y_pos, current_y_pos],
                                mode='lines',
                                line=dict(color=colors['delay'], width=22),
                                name=f"Retard: {task.name}",
                                showlegend=False,
                                hovertemplate=hover_text + f"<br><b>Retard: {expected_progress - task.progress:.1f}%</b><extra></extra>"
                            ))
            
            # 1.4. Indicateur de chemin critique
            if task.id in critical_path:
                fig.add_trace(go.Scatter(
                    x=[task.start_date - timedelta(days=1)],
                    y=[current_y_pos],
                    mode='markers',
                    marker=dict(symbol='star', size=15, color=colors['critical_path'], 
                              line=dict(width=2, color='#F57F17')),
                    name=f"Critique: {task.name}",
                    showlegend=False,
                    hovertemplate=hover_text + "<br><b>Chemin critique</b><extra></extra>"
                ))
            
            # 1.5. Zone d'incertitude
            if show_uncertainty and task.get_standard_deviation() and task.get_standard_deviation() > 0:
                cumulative_uncertainty = calculate_cumulative_uncertainty(tm, task.id)
                uncertainty_end = end_date + timedelta(days=cumulative_uncertainty)
                
                if uncertainty_end > end_date:
                    # Zone d'incertitude
                    fig.add_trace(go.Scatter(
                        x=[end_date, uncertainty_end, uncertainty_end, end_date, end_date],
                        y=[current_y_pos - 0.1, current_y_pos - 0.1, current_y_pos + 0.1, current_y_pos + 0.1, current_y_pos - 0.1],
                        fill="toself",
                        fillcolor='rgba(156, 39, 176, 0.2)',
                        line=dict(color=colors['uncertainty'], width=1, dash='dot'),
                        name=f"Incertitude: {task.name}",
                        showlegend=False,
                        hovertemplate=hover_text + f"<br><b>Incertitude cumulée: ±{cumulative_uncertainty} jours</b><extra></extra>"
                    ))
            
            # --- NIVEAU 2: TENDANCES (Y = current_y_pos - trend_height) ---
            trend_y = current_y_pos - trend_height
            
            if show_trend and task.progress > 0 and task.progress < 100:
                optimistic_end, pessimistic_end, trend = calculate_trend_projection(task, today)
                
                if optimistic_end and pessimistic_end:
                    # Ligne de tendance optimiste (triangle vers le haut)
                    fig.add_trace(go.Scatter(
                        x=[today, optimistic_end],
                        y=[trend_y, trend_y],
                        mode='lines+markers',
                        line=dict(color=colors['trend_accelerating'], width=4),
                        marker=dict(symbol='triangle-up', size=10, color=colors['trend_accelerating']),
                        name=f"Tendance optimiste: {task.name}",
                        showlegend=False,
                        hovertemplate=f"<b>SCÉNARIO OPTIMISTE</b><br>{optimistic_end.strftime('%Y-%m-%d')}<extra></extra>"
                    ))
                    
                    # Ligne de tendance pessimiste (triangle vers le bas)
                    fig.add_trace(go.Scatter(
                        x=[today, pessimistic_end],
                        y=[trend_y, trend_y],
                        mode='lines+markers',
                        line=dict(color=colors['trend_decelerating'], width=4),
                        marker=dict(symbol='triangle-down', size=10, color=colors['trend_decelerating']),
                        name=f"Tendance pessimiste: {task.name}",
                        showlegend=False,
                        hovertemplate=f"<b>SCÉNARIO PESSIMISTE</b><br>{pessimistic_end.strftime('%Y-%m-%d')}<extra></extra>"
                    ))
                    
                    # Indicateur de tendance central
                    trend_color = colors['trend_accelerating'] if trend == 'accelerating' else \
                                 colors['trend_decelerating'] if trend == 'decelerating' else colors['dependencies']
                    trend_symbol = 'triangle-up' if trend == 'accelerating' else \
                                  'triangle-down' if trend == 'decelerating' else 'diamond'
                    
                    fig.add_trace(go.Scatter(
                        x=[today],
                        y=[trend_y],
                        mode='markers',
                        marker=dict(symbol=trend_symbol, size=15, color=trend_color, 
                                  line=dict(width=2, color='white')),
                        name=f"Tendance: {task.name}",
                        showlegend=False,
                        hovertemplate=f"<b>TENDANCE: {trend.upper()}</b><extra></extra>"
                    ))
            
            current_y_pos += 1
    
    # --- DÉPENDANCES ---
    if show_dependencies:
        for task in tm.tasks.values():
            if task.id not in task_positions:
                continue
                
            for dep_id in task.get_dependency_ids():
                if dep_id not in task_positions:
                    continue
                    
                dep_task = tm.tasks.get(dep_id)
                if dep_task and dep_task.get_stochastic_end_date() and task.start_date:
                    # Flèche de dépendance plus visible
                    fig.add_annotation(
                        x=task.start_date,
                        y=task_positions[task.id],
                        ax=dep_task.get_stochastic_end_date(),
                        ay=task_positions[dep_id],
                        xref='x', yref='y',
                        axref='x', ayref='y',
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1,
                        arrowwidth=2,
                        arrowcolor=colors['dependencies'],
                        opacity=0.7
                    )
    
    # --- LIGNE DU JOUR ---
    fig.add_shape(
        type="line",
        x0=today, x1=today,
        y0=-0.8, y1=current_y_pos-0.5,
        line=dict(color=colors['today_line'], width=3, dash="solid"),
    )
    
    # Annotation ligne du jour
    fig.add_annotation(
        x=today,
        y=current_y_pos-0.5,
        text=f"📅 {today.strftime('%d/%m/%Y')}",
        showarrow=True,
        arrowhead=2,
        ax=0,
        ay=-30,
        bgcolor=colors['today_line'],
        font=dict(color="white", size=12, family="Arial Black"),
        bordercolor="white",
        borderwidth=1
    )
    
    # --- SÉPARATEURS DE GROUPE ---
    for separator_y in group_separators:
        fig.add_shape(
            type="line",
            x0=zoom_start or min(task.start_date for task in tm.tasks.values() if task.start_date),
            x1=zoom_end or max(task.get_stochastic_end_date() or task.start_date for task in tm.tasks.values() if task.start_date),
            y0=separator_y,
            y1=separator_y,
            line=dict(color="#BDBDBD", width=2, dash="dot"),
        )
    
    # --- CONFIGURATION DU LAYOUT ---
    fig.update_layout(
        title=dict(
            text="📊 Diagramme de Gantt Avancé - Planification & Tendances",
            font=dict(size=18, color='#1A237E', family="Arial Black"),
            x=0.5
        ),
        xaxis=dict(
            title="📅 Chronologie",
            type='date',
            tickformat='%d/%m',
            range=[zoom_start, zoom_end] if zoom_start else None,
            gridcolor='#F5F5F5',
            showgrid=True
        ),
        yaxis=dict(
            title="",
            tickmode='array',
            tickvals=list(range(len(y_labels))),
            ticktext=y_labels,
            autorange='reversed',
            gridcolor='#F5F5F5',
            showgrid=True
        ),
        height=chart_height,
        hovermode='closest',
        showlegend=False,
        plot_bgcolor='#FAFAFA',
        paper_bgcolor='white',
        margin=dict(l=200, r=50, t=80, b=50)
    )
    
    # Afficher le graphique
    st.plotly_chart(fig, use_container_width=True)
    
    # --- LÉGENDE ADAPTÉE ---
    st.markdown("#### 📋 Légende")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 📅 Planification")
        st.markdown("- Gris: Plan prévu")
        st.markdown("- Vert: Progression")
        st.markdown("- Rouge: Retard")
        st.markdown("- Or: Tâche critique")
    
    with col2:
        st.markdown("#### 📈 Tendances")
        st.markdown("- Triangle vert: Accélération")
        st.markdown("- Triangle jaune: Ralentissement")
        st.markdown("- Losange: Stabilité")
        st.markdown("- Ligne rose: Aujourd'hui")
    
    with col3:
        st.markdown("#### 🔍 Incertitudes")
        st.markdown("- Zone violette: Marge d'incertitude")
        st.markdown("- Flèches: Dépendances")