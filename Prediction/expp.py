# =============================================================================
#                              IMPORTS
# =============================================================================
# Mettons ici tous les outils dont on aura besoin
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # Pour de jolis graphiques
import joblib          # Pratique pour sauvegarder/charger notre modèle
import json            # Pour gérer les listes d'options au format texte
import os              # Pour vérifier si un fichier existe

# Les outils spécifiques au Machine Learning de Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier # L'Arbre de Décision
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline as SklearnPipeline 


from imblearn.pipeline import Pipeline as ImblearnPipeline 
from imblearn.over_sampling import SMOTE                 

print("Les bibliothèques sont prêtes (y compris pour l'Arbre de Décision) !")

print("\n--- Paramètres de notre entraînement ---")

CSV_FILE_PATH = "C:/Users/feres/Downloads/CANEVAS_RETAILS.csv" 

PIPELINE_FOR_STREAMLIT_APP = 'random_forest_credit_pipeline.joblib' 

PIPELINE_DT_SAVE_NAME = 'decision_tree_credit_pipeline.joblib' 
OPTIONS_FILENAME = 'categorical_options.json'              

# --- Noms des Colonnes ---
# On définit les noms de colonnes qu'on s'attend à trouver
EXPECTED_COLUMNS = [
    'Unnamed: 0', 
    'Agence Initiatrice', 'N°CANEVAS', 'Type CANEVAS', 'Date Création', 
    'Code Produit', 'Montant Sollicité', 'Revenus Annuel', 'Retenus Mensuel', 
    'Profession', 'Date Naissance', 'Sexe', 'Décision Finale', 
    'Centre Décision', 'Date MEP'
] 
TARGET_COL = 'Décision Finale'  
ID_COL = 'N°CANEVAS'            

NUMERICAL_COLS_BASE = ['Montant Sollicité', 'Revenus Annuel', 'Retenus Mensuel'] 
DATE_COLS = ['Date Création', 'Date Naissance', 'Date MEP'] # Les colonnes qui contiennent des dates

# --- Paramètres pour l'entraînement ---
TEST_SET_SIZE = 0.2  # On garde 20% des données de côté pour tester
RANDOM_STATE = 42    # Pour que les résultats soient les mêmes si on relance
N_ESTIMATORS_RF = 100   # Le nombre d'arbres dans notre Random Forest

print(f"Données source : {CSV_FILE_PATH}")
print(f"Objectif : Prédire '{TARGET_COL}'")
print(f"Pipeline pour Streamlit (Random Forest) sera sauvegardé dans : {PIPELINE_FOR_STREAMLIT_APP}")
print(f"Pipeline Arbre de Décision sera sauvegardé dans : {PIPELINE_DT_SAVE_NAME}")

print(f"\n--- Chargement du fichier CSV : {CSV_FILE_PATH} ---")
try:
    df = pd.read_csv(CSV_FILE_PATH, sep=',', quotechar='"', encoding='utf-8', skiprows=3, header=0)
    print("Lecture  réussie.")
except UnicodeDecodeError:
    print("Erreur avec UTF-8, on tente latin-1...")
    try:
        df = pd.read_csv(CSV_FILE_PATH, sep=',', quotechar='"', encoding='latin-1', skiprows=3, header=0)
        print("Lecture en latin-1 réussie.")
    except Exception as e:
        print(f"ERREUR FATALE: Impossible de lire le fichier CSV : {e}")
        exit() 
except FileNotFoundError:
    print(f"ERREUR FATALE: Le fichier '{CSV_FILE_PATH}' est introuvable !")
    exit()
except Exception as e:
    print(f"ERREUR FATALE: Problème inattendu lors de la lecture : {e}")
    exit()
print(f"Nombre de lignes chargées : {len(df)}")

print("\n--- Nettoyage et renommage des colonnes ---")
if len(df.columns) == len(EXPECTED_COLUMNS) -1 and 'Unnamed: 0' not in df.columns:
     print("Ajustement des colonnes attendues (retrait de 'Unnamed: 0').")
     EXPECTED_COLUMNS = [col for col in EXPECTED_COLUMNS if col != 'Unnamed: 0']
elif len(df.columns) == len(EXPECTED_COLUMNS) +1 and 'Unnamed: 0' in df.columns and 'Unnamed: 0' not in EXPECTED_COLUMNS:
      print("Ajustement des colonnes attendues (ajout de 'Unnamed: 0').")
      EXPECTED_COLUMNS.insert(0,'Unnamed: 0')

if len(df.columns) == len(EXPECTED_COLUMNS):
    print("Renommage des colonnes...")
    df.columns = EXPECTED_COLUMNS
    if 'Unnamed: 0' in df.columns:
        print("Suppression de la colonne 'Unnamed: 0'.")
        df = df.drop(columns=['Unnamed: 0'])
else:
    print(f"ALERTE: Le nombre de colonnes ({len(df.columns)}) ne correspond pas à attendu ({len(EXPECTED_COLUMNS)}).")
    print("On continue avec les noms lus. Vérifiez la config.")
print(f"Noms des colonnes finaux : {df.columns.tolist()}")

print("\n--- Nettoyage avancé et création de nouvelles variables ---")
print("Nettoyage des colonnes numériques...")
for col in NUMERICAL_COLS_BASE:
    if col in df.columns:
        if df[col].dtype == 'object': 
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '', regex=False), errors='coerce')
        elif not pd.api.types.is_numeric_dtype(df[col]):
             df[col] = pd.to_numeric(df[col], errors='coerce')
             print(f"  -> Colonne '{col}' convertie en nombre.")

print("Traitement des dates et calcul de l'âge...")
for col in DATE_COLS:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], format='%d/%m/%Y', errors='coerce')

numerical_cols = NUMERICAL_COLS_BASE.copy() 
if 'Date Création' in df.columns and 'Date Naissance' in df.columns and \
   pd.api.types.is_datetime64_any_dtype(df['Date Création']) and \
   pd.api.types.is_datetime64_any_dtype(df['Date Naissance']):
    df['Age'] = (df['Date Création'] - df['Date Naissance']).dt.days / 365.25
    df.loc[df['Age'] < 18, 'Age'] = np.nan 
    df.loc[df['Age'] > 100, 'Age'] = np.nan 
    numerical_cols.append('Age')
    print("Colonne 'Age' créée.")
else:
    print("Impossible de créer 'Age'.")

print("Nettoyage des colonnes texte...")
string_cols = df.select_dtypes(include='object').columns
for col in string_cols:
    if col in df.columns: 
        df[col] = df[col].str.strip()

# --- Matrice de Corrélation ---
print("\n--- Analyse : Matrice de Corrélation des Variables Numériques ---")
cols_for_corr = [col for col in numerical_cols if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
if len(cols_for_corr) > 1:
    correlation_matrix = df[cols_for_corr].corr()
    print("Matrice de Corrélation calculée.")
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Matrice de Corrélation des Variables Numériques')
    plt.tight_layout()
    plt.show() 
else:
    print("Pas assez de variables numériques pour une matrice de corrélation.")

# =============================================================================
#              PRÉPARATION DES DONNÉES POUR LE MODÈLE
# =============================================================================
print(f"\n--- Préparation des données pour le modèle (Cible : {TARGET_COL}) ---")
initial_rows = len(df)
df.dropna(subset=[TARGET_COL], inplace=True)
if initial_rows - len(df) > 0:
    print(f"Suppression de {initial_rows - len(df)} lignes sans '{TARGET_COL}'.")
if df.empty:
    print("ERREUR FATALE: Plus aucune donnée !")
    exit()
print(f"Nombre de lignes utilisables : {len(df)}")

potential_cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
cat_cols = [col for col in potential_cat_cols if col not in [TARGET_COL, ID_COL]] 
features = numerical_cols + cat_cols
features = [f for f in features if f in df.columns] 
print(f"\nFeatures numériques retenues : {numerical_cols}")
print(f"Features catégorielles retenues : {cat_cols}")
print(f"\nListe finale des features ({len(features)} colonnes) : {features}")
if not features:
     print("ERREUR FATALE: Aucune feature sélectionnée !")
     exit()

X = df[features]
y = df[TARGET_COL]
print(f"\nRépartition de la cible (avant rééquilibrage) :\n{y.value_counts(normalize=True)}") 

# =============================================================================
#                  SÉPARATION EN JEUX D'ENTRAÎNEMENT ET DE TEST
# =============================================================================
print(f"\n--- Division des données : {100-TEST_SET_SIZE*100}% entraînement, {TEST_SET_SIZE*100}% test ---")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SET_SIZE, random_state=RANDOM_STATE, stratify=y  
)
print(f"Taille jeu entraînement : {X_train.shape}, Taille jeu test : {X_test.shape}")

# =============================================================================
#                 MISE EN PLACE DU PRÉTRAITEMENT AUTOMATIQUE
# =============================================================================
print("\n--- Préparation du pipeline de prétraitement ---")
numerical_features_in_train = X_train.select_dtypes(include=np.number).columns.tolist()
categorical_features_in_train = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

print(f"Colonnes numériques à prétraiter : {numerical_features_in_train}")
print(f"Colonnes catégorielles à prétraiter : {categorical_features_in_train}")

numerical_transformer = SklearnPipeline(steps=[('imputer', SimpleImputer(strategy='median'))])
categorical_transformer = SklearnPipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features_in_train), 
        ('cat', categorical_transformer, categorical_features_in_train) 
    ],
    remainder='passthrough' 
)
print("Préprocesseur combiné prêt.")

# =============================================================================
#           CRÉATION, ENTRAÎNEMENT ET ÉVALUATION DES MODÈLES
# =============================================================================
# On stockera les résultats ici
model_accuracies = {}

print("\n" + "="*30 + " MODÈLE 1: RANDOM FOREST " + "="*30)
rf_pipeline_model = ImblearnPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=RANDOM_STATE)),
    ('classifier', RandomForestClassifier(n_estimators=N_ESTIMATORS_RF, random_state=RANDOM_STATE, n_jobs=-1))
])
print("Entraînement du Random Forest...")
rf_pipeline_model.fit(X_train, y_train)
print("Random Forest entraîné !")
y_pred_rf = rf_pipeline_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
model_accuracies['Random Forest'] = accuracy_rf
print(f"\n--- Évaluation du Random Forest ---")
print(f"Accuracy (Random Forest) : {accuracy_rf:.4f} ({accuracy_rf*100:.2f}%)")
print("Rapport de Classification (Random Forest) :")
print(classification_report(y_test, y_pred_rf, zero_division=0))
print("\nMatrice de Confusion (Random Forest)...")
try:
    cm_rf = confusion_matrix(y_test, y_pred_rf, labels=rf_pipeline_model.classes_) 
    disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=rf_pipeline_model.classes_)
    fig_rf, ax_rf = plt.subplots(figsize=(6, 5)) 
    disp_rf.plot(cmap='Greens', ax=ax_rf, colorbar=False) 
    plt.title('Matrice de Confusion (Random Forest)')
    plt.show() 
except Exception as e_rf:
    print(f"Erreur affichage matrice confusion RF: {e_rf}")


print("\n" + "="*30 + " MODÈLE 2: ARBRE DE DÉCISION " + "="*30)
dt_pipeline_model = ImblearnPipeline(steps=[
    ('preprocessor', preprocessor), 
    ('smote', SMOTE(random_state=RANDOM_STATE)), 
    ('classifier', DecisionTreeClassifier(random_state=RANDOM_STATE))
])
print("Entraînement de l'Arbre de Décision...")
dt_pipeline_model.fit(X_train, y_train)
print("Arbre de Décision entraîné !")
y_pred_dt = dt_pipeline_model.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
model_accuracies['Arbre de Décision'] = accuracy_dt
print(f"\n--- Évaluation de l'Arbre de Décision ---")
print(f"Accuracy (Arbre de Décision) : {accuracy_dt:.4f} ({accuracy_dt*100:.2f}%)")
print("Rapport de Classification (Arbre de Décision) :")
print(classification_report(y_test, y_pred_dt, zero_division=0))
print("\nMatrice de Confusion (Arbre de Décision)...")
try:
    cm_dt = confusion_matrix(y_test, y_pred_dt, labels=dt_pipeline_model.classes_) 
    disp_dt = ConfusionMatrixDisplay(confusion_matrix=cm_dt, display_labels=dt_pipeline_model.classes_)
    fig_dt, ax_dt = plt.subplots(figsize=(6, 5)) 
    disp_dt.plot(cmap='Oranges', ax=ax_dt, colorbar=False) 
    plt.title('Matrice de Confusion (Arbre de Décision)')
    plt.show() 
except Exception as e_dt:
    print(f"Erreur affichage matrice confusion DT: {e_dt}")

# =============================================================================
#            COMPARAISON ET SAUVEGARDE DES MODÈLES
# =============================================================================
print("\n" + "="*30 + " BILAN DES PERFORMANCES " + "="*30)
print("Accuracies obtenues :")
for model_name, acc in model_accuracies.items():
    print(f"- {model_name:<20} : {acc:.4f}")

# Sauvegarde du pipeline Random Forest (pour l'application Streamlit)
print(f"\n--- Sauvegarde du pipeline Random Forest pour l'application ---")
print(f"Le pipeline Random Forest sera sauvegardé dans : {PIPELINE_FOR_STREAMLIT_APP}")
try:
    joblib.dump(rf_pipeline_model, PIPELINE_FOR_STREAMLIT_APP) 
    print(f"Pipeline RF sauvegardé dans '{PIPELINE_FOR_STREAMLIT_APP}'.")
except Exception as e:
    print(f"ERREUR de sauvegarde du pipeline RF : {e}")

# Sauvegarde optionnelle du pipeline Arbre de Décision
print(f"\n--- Sauvegarde du pipeline Arbre de Décision (pour référence) ---")
print(f"Le pipeline Arbre de Décision sera sauvegardé dans : {PIPELINE_DT_SAVE_NAME}")
try:
    joblib.dump(dt_pipeline_model, PIPELINE_DT_SAVE_NAME)
    print(f"Pipeline DT sauvegardé dans '{PIPELINE_DT_SAVE_NAME}'.")
except Exception as e:
    print(f"ERREUR de sauvegarde du pipeline DT : {e}")

# Sauvegarde des options pour les menus déroulants
print("\nSauvegarde des options pour l'interface...")
options_dict = {}
# Utiliser categorical_features_in_train car c'est ce que le préprocesseur connaît
for col in categorical_features_in_train: 
    if col in df.columns: # Prendre les options depuis le df original (plus complet)
        unique_values = df[col].dropna().unique().tolist()
        try:
            options_dict[col] = sorted([str(val) for val in unique_values]) 
        except TypeError: 
            options_dict[col] = [str(val) for val in unique_values]
print(f"Sauvegarde des options dans : {OPTIONS_FILENAME}")
try:
    with open(OPTIONS_FILENAME, 'w', encoding='utf-8') as f:
        json.dump(options_dict, f, ensure_ascii=False, indent=4)
    print("Options sauvegardées.")
except Exception as e:
    print(f"ERREUR de sauvegarde des options : {e}")

# =============================================================================
#                                 FIN
# =============================================================================
print("\n--- Script d'entraînement des deux modèles terminé ! ---")