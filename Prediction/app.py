import streamlit as st
import pandas as pd
import numpy as np
import joblib  
import json    
import os      
st.set_page_config(page_title="Prédiction Crédit (Random Forest)", layout="wide")
PIPELINE_FILENAME = 'random_forest_credit_pipeline.joblib' 
OPTIONS_FILENAME = 'categorical_options.json'           

@st.cache_resource 
def load_pipeline(filename):
    """Charge le pipeline de modèle depuis un fichier."""
    if not os.path.exists(filename):
        st.error(f"Fichier modèle '{filename}' absent. Il faut lancer l'entraînement d'abord.")
        return None
    try:
        pipeline = joblib.load(filename)
        print(f"Modèle '{filename}' chargé avec succès.") # Message dans le terminal
        return pipeline
    except Exception as e:
        st.error(f"Problème lors du chargement du modèle '{filename}': {e}")
        return None

@st.cache_data # Pour les données plus simples comme notre liste d'options
def load_options(filename):
    """Charge les options des menus déroulants depuis un fichier JSON."""
    if not os.path.exists(filename):
        st.error(f"Fichier d'options '{filename}' absent. L'entraînement a-t-il bien généré ce fichier ?")
        return None
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            options = json.load(f)
        print(f"Options '{filename}' chargées.") # Message dans le terminal
        return options
    except Exception as e:
        st.error(f"Impossible de lire les options '{filename}': {e}")
        return None

rf_pipeline = load_pipeline(PIPELINE_FILENAME) # On charge le pipeline Random Forest
categorical_options = load_options(OPTIONS_FILENAME)


st.title("Simulateur de Décision de Crédit ")
st.markdown("Remplissez les informations du demandeur pour obtenir une estimation.")

if rf_pipeline is None or categorical_options is None:
    st.warning("L'application ne peut démarrer car des fichiers essentiels manquent. Vérifiez les erreurs ci-dessus.")
    st.stop() 

st.sidebar.header("Infos Demandeur")
input_data = {}

try:
    print("Extraction des features attendues par le pipeline...")
    num_features = rf_pipeline.named_steps['preprocessor'].transformers_[0][2]
    cat_features = rf_pipeline.named_steps['preprocessor'].transformers_[1][2]
    expected_features = num_features + cat_features
    print(f"--> Features nécessaires (détectées par le pipeline RF) : {expected_features}") 
except Exception as e:
    st.warning(f"Impossible de lire les features du modèle ({e}). Utilisation d'une liste par défaut.")

    num_features = ['Montant Sollicité', 'Revenus Annuel', 'Retenus Mensuel', 'Age']
    cat_features = ['Type CANEVAS', 'Profession', 'Sexe', 'Centre Décision'] 
    expected_features = num_features + cat_features
    print(f"--> Features nécessaires (manuelles) : {expected_features}")

with st.sidebar:
    st.subheader("Données Financières et Âge")
    default_montant = 5000000.0
    default_revenu = 20000000.0
    default_retenu = 0.0
    default_age = 35.0 
    for feature in num_features: 
        if feature == 'Age':
             input_data[feature] = st.number_input(f"Âge", min_value=18.0, max_value=100.0, value=default_age, step=1.0)
        elif feature == 'Montant Sollicité':
             input_data[feature] = st.number_input(f"Montant demandé", min_value=0.0, value=default_montant, step=100000.0, format="%.2f")
        elif feature == 'Revenus Annuel':
             input_data[feature] = st.number_input(f"Revenu annuel", min_value=0.0, value=default_revenu, step=100000.0, format="%.2f")
        elif feature == 'Retenus Mensuel':
             input_data[feature] = st.number_input(f"Charges mensuelles", min_value=0.0, value=default_retenu, step=50000.0, format="%.2f")

    st.subheader("Autres Détails")
    for feature in cat_features:
        options = categorical_options.get(feature, []) 
        if not options:
             st.warning(f"Pas d'options pour '{feature}'. Saisie manuelle.")
             input_data[feature] = st.text_input(f"Entrez {feature}", value="") 
        else:
            input_data[feature] = st.selectbox(
                f"{feature.replace('_', ' ')}", 
                options=options, 
                index=0 
            )

col1, col2 = st.columns([2, 3]) 

with col1: 
    st.subheader("Lancer la Prédiction")
    if st.button(" Prédire la Décision", use_container_width=True, type="primary"):

        if any(v is None or (isinstance(v, str) and v.strip() == "") for k, v in input_data.items()):
             st.error("Merci de remplir toutes les informations demandées.")
        else:

            try:
                input_df = pd.DataFrame([input_data], columns=expected_features)
                st.write("--- Données fournies ---")
                df_display_formatted = input_df.copy()
                for num_col_format in num_features:
                    if num_col_format in df_display_formatted.columns:
                        df_display_formatted[num_col_format] = df_display_formatted[num_col_format].map('{:,.2f}'.format)
                st.dataframe(df_display_formatted)

                prediction = rf_pipeline.predict(input_df) 
                try:
                    probabilities = rf_pipeline.predict_proba(input_df)
                    probability_df = pd.DataFrame(probabilities, columns=rf_pipeline.classes_, index=["Probabilité"])
                except AttributeError: 
                     probabilities = None
                     probability_df = pd.DataFrame({"Info": ["Probabilités non disponibles"]})

                st.write("--- Résultat de la Simulation (Random Forest) ---")
                predicted_class = prediction[0] 
                if predicted_class == 'ACCORD': 
                    st.success(f" Prédiction : **{predicted_class}**")
                else: 
                    st.error(f"Prédiction : **{predicted_class}**")

                st.write("Probabilités estimées :")
                st.dataframe(probability_df.style.format("{:.1%}")) 

            except Exception as e:
                st.error(f"Une erreur s'est produite pendant l'analyse :")
                st.exception(e) 
                st.error("Vérifiez les données ou réessayez.")
