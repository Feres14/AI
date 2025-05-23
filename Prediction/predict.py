import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  
import joblib          
import json            
import os             
import io              

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline as SklearnPipeline 


from imblearn.pipeline import Pipeline as ImblearnPipeline 
from imblearn.over_sampling import SMOTE                 

print("Les bibliothèques sont prêtes !")

# =============================================================================
#                            CONFIGURATION
# =============================================================================
print("\n--- Paramètres de notre entraînement ---")
# --- Fichiers ---
CSV_FILE_PATH = "C:/Users/feres/Downloads/CANEVAS_RETAILS.csv" # Le chemin vers vos données brutes
PIPELINE_FILENAME = 'random_forest_credit_pipeline.joblib'  # Nom du fichier principal pour le pipeline RF (utilisé par Streamlit)
PIPELINE_DT_SAVE_NAME = 'decision_tree_credit_pipeline.joblib' # Nom pour sauvegarder le modèle Arbre de Décision
OPTIONS_FILENAME = 'categorical_options.json'               # Nom du fichier pour les options des menus déroulants

# --- Noms des Colonnes ---
EXPECTED_COLUMNS = [
    'Unnamed: 0', 'Agence Initiatrice', 'N°CANEVAS', 'Type CANEVAS', 'Date Création', 
    'Code Produit', 'Montant Sollicité', 'Revenus Annuel', 'Retenus Mensuel', 
    'Profession', 'Date Naissance', 'Sexe', 'Décision Finale', 
    'Centre Décision', 'Date MEP'
] 
TARGET_COL = 'Décision Finale'
ID_COL = 'N°CANEVAS'            
NUMERICAL_COLS_BASE = ['Montant Sollicité', 'Revenus Annuel', 'Retenus Mensuel'] 
DATE_COLS = ['Date Création', 'Date Naissance', 'Date MEP']

# --- Paramètres pour l'entraînement ---
TEST_SET_SIZE = 0.2
RANDOM_STATE = 42   
N_ESTIMATORS_RF = 100

print(f"Données source : {CSV_FILE_PATH}")
print(f"Objectif : Prédire '{TARGET_COL}'")
print(f"Pipeline pour Streamlit (Random Forest) sera sauvegardé dans : {PIPELINE_FILENAME}") 
print(f"Pipeline Arbre de Décision sera sauvegardé dans : {PIPELINE_DT_SAVE_NAME}")

# =============================================================================
#                    CHARGEMENT DES DONNÉES
# =============================================================================
print(f"\n--- Étape 1 : Chargement du fichier de données ---")
print(f"Source : {CSV_FILE_PATH}")
try:
    df_initial = pd.read_csv(CSV_FILE_PATH, sep=',', quotechar='"', encoding='utf-8', skiprows=3, header=0)
    print("Fichier de données chargé avec succes ")
except UnicodeDecodeError:
    print("Échec avec UTF-8, tentative avec latin-1...")
    try:
        df_initial = pd.read_csv(CSV_FILE_PATH, sep=',', quotechar='"', encoding='latin-1', skiprows=3, header=0)
        print("Fichier de données chargé avec succès (encodage latin-1).")
    except Exception as e:
        print(f"ERREUR FATALE: Impossible de lire le fichier CSV : {e}")
        exit() 
except FileNotFoundError:
    print(f"ERREUR FATALE: Le fichier '{CSV_FILE_PATH}' est introuvable !")
    exit()
except Exception as e:
    print(f"ERREUR FATALE: Un problème inattendu est survenu lors de la lecture du fichier : {e}")
    exit()

df = df_initial.copy()
print(f"Nombre total de lignes initialement chargées : {len(df)}")

# =============================================================================
#      DESCRIPTION ET EXPLORATION INITIALE DES DONNÉES SOURCES
# =============================================================================
print("\n--- Étape 2 : Description et Exploration Initiale des Données ---")

# --- 2.1 Renommage et premier aperçu ---
print("\nApplication des noms de colonnes et aperçu (df.head()) :")
if len(df.columns) == len(EXPECTED_COLUMNS) -1 and 'Unnamed: 0' not in df.columns:
     print("  Ajustement des colonnes attendues (retrait de 'Unnamed: 0').")
     EXPECTED_COLUMNS = [col for col in EXPECTED_COLUMNS if col != 'Unnamed: 0']
elif len(df.columns) == len(EXPECTED_COLUMNS) +1 and 'Unnamed: 0' in df.columns and 'Unnamed: 0' not in EXPECTED_COLUMNS:
      print("  Ajustement des colonnes attendues (ajout de 'Unnamed: 0').")
      EXPECTED_COLUMNS.insert(0,'Unnamed: 0')

if len(df.columns) == len(EXPECTED_COLUMNS):
    df.columns = EXPECTED_COLUMNS
    if 'Unnamed: 0' in df.columns: 
        print("  Suppression de la colonne 'Unnamed: 0' (index inutile).")
        df = df.drop(columns=['Unnamed: 0'])
    print("  Noms de colonnes appliqués.")
else:
    print(f"  ALERTE: Le nombre de colonnes lues ({len(df.columns)}) ne correspond pas au nombre attendu ({len(EXPECTED_COLUMNS)}).")
    print("  Les noms de colonnes bruts seront utilisés pour la suite de cette section. Vérifiez 'EXPECTED_COLUMNS'.")
print(df.head())
print(f"\nDimensions du jeu de données après ajustement initial : {df.shape} (lignes, colonnes)")
print(f"Liste des colonnes : {df.columns.tolist()}")

# --- 2.2 Description détaillée (df.info()) ---
print("\nDescription détaillée des colonnes (types, valeurs non nulles - df.info()) :")
buffer = io.StringIO()
df.info(buf=buffer)
info_str = buffer.getvalue()
print(info_str)

# --- 2.3 Statistiques descriptives (df.describe()) ---
print("\nStatistiques descriptives pour les colonnes numériques (df.describe()) :")
df_temp_describe = df.copy()
potential_num_for_describe = ['Montant Sollicité', 'Revenus Annuel', 'Retenus Mensuel', 'Agence Initiatrice', 'Code Produit']
for col in potential_num_for_describe:
    if col in df_temp_describe.columns and df_temp_describe[col].dtype == 'object':
        try:
            df_temp_describe[col] = pd.to_numeric(df_temp_describe[col].astype(str).str.replace(',', '', regex=False), errors='coerce')
        except: pass 
numeric_cols_for_describe = df_temp_describe.select_dtypes(include=np.number).columns.tolist()
if numeric_cols_for_describe:
    print(df_temp_describe[numeric_cols_for_describe].describe().transpose())
else:
    print("  Aucune colonne numérique trouvée pour les statistiques descriptives à ce stade.")


#  Distribution de la variable cible 
if TARGET_COL in df.columns:
    print(f"\n Analyse : Distribution de la Variable Cible ('{TARGET_COL}') ")
    target_counts = df[TARGET_COL].value_counts()
    target_proportions = df[TARGET_COL].value_counts(normalize=True) * 100
    print("Distribution des valeurs :")
    print(target_counts)
    print("\nEn pourcentage :")
    print(target_proportions)

    print("\n Visualisation de la distribution de la variable cible :")
    plt.figure(figsize=(8, 6)) 
    sns.set_style('darkgrid') 
    
    ax = sns.countplot(x=TARGET_COL, data=df, palette='winter', order=target_counts.index) 
    
    plt.title(f'Distribution de la Décision Finale ({TARGET_COL})', fontsize=15)
    plt.xlabel('Décision Finale', fontsize=12)
    plt.ylabel('Nombre d\'occurrences (Count)', fontsize=12)
    
    total = len(df[TARGET_COL])
    if total > 0: 
        for p in ax.patches:
            percentage = f'{100 * p.get_height() / total:.1f}%'
            x = p.get_x() + p.get_width() / 2
            y = p.get_y() + p.get_height() + (total*0.005) 
            ax.text(x, y, percentage, ha='center', va='bottom', fontsize=10)
            
    plt.tight_layout() 
    plt.show()
else:
    print(f"ALERTE: La colonne cible '{TARGET_COL}' est introuvable. Impossible de visualiser sa distribution.")

# =============================================================================
#                NETTOYAGE DÉTAILLÉ & FEATURE ENGINEERING
# =============================================================================
print("\n--- Étape 3 : Nettoyage détaillé des données et création de variables ---")
print("Nettoyage des colonnes numériques (celles utilisées comme features)...")
for col in NUMERICAL_COLS_BASE: 
    if col in df.columns:
        if df[col].dtype == 'object': 
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '', regex=False), errors='coerce')
        elif not pd.api.types.is_numeric_dtype(df[col]): 
             df[col] = pd.to_numeric(df[col], errors='coerce')
             print(f"  -> Colonne '{col}' (feature) convertie en type numérique standard.")

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
    print("Colonne 'Age' créée et ajoutée aux features numériques.")
else:
    print("Impossible de créer 'Age' (vérifiez les colonnes 'Date Création' et 'Date Naissance').")

print("Nettoyage des colonnes texte (suppression des espaces en trop)...")
string_cols = df.select_dtypes(include='object').columns
for col in string_cols:
    if col in df.columns: 
        df[col] = df[col].str.strip()

print("\n--- Analyse : Matrice de Corrélation des Variables Numériques (features) ---")
cols_for_corr_final = [col for col in numerical_cols if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
if len(cols_for_corr_final) > 1:
    correlation_matrix = df[cols_for_corr_final].corr()
    print("Matrice de Corrélation calculée sur les features numériques finales.")
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Matrice de Corrélation des Features Numériques Finales')
    plt.tight_layout()
    plt.show() 
else:
    print("Pas assez de features numériques finales pour une matrice de corrélation.")

# =============================================================================
#              PRÉPARATION DES DONNÉES POUR LE MODÈLE
# =============================================================================
print(f"\n--- Étape 4 : Préparation des données pour les modèles (Cible : {TARGET_COL}) ---")
initial_rows_before_target_dropna = len(df)
df.dropna(subset=[TARGET_COL], inplace=True) 
if initial_rows_before_target_dropna - len(df) > 0:
    print(f"Suppression de {initial_rows_before_target_dropna - len(df)} lignes car la cible ('{TARGET_COL}') était manquante.")
if df.empty:
    print("ERREUR FATALE: Plus aucune donnée après suppression des cibles manquantes !")
    exit()
print(f"Nombre de lignes prêtes pour la modélisation : {len(df)}")

potential_cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
cat_cols = [col for col in potential_cat_cols if col not in [TARGET_COL, ID_COL, 'Agence Initiatrice', 'Code Produit']] 
features = numerical_cols + cat_cols
features = [f for f in features if f in df.columns] 
print(f"\nFeatures numériques utilisées pour l'entraînement : {numerical_cols}")
print(f"Features catégorielles utilisées pour l'entraînement : {cat_cols}")
print(f"\nListe complète des features ({len(features)} colonnes) : {features}")
if not features:
     print("ERREUR FATALE: Aucune feature n'a été sélectionnée pour l'entraînement !")
     exit()

X = df[features]
y = df[TARGET_COL]
print(f"\nDistribution de la variable cible (avant rééquilibrage) :\n{y.value_counts(normalize=True)}") 

# =============================================================================
#                  SÉPARATION EN JEUX D'ENTRAÎNEMENT ET DE TEST
# =============================================================================
print(f"\n--- Étape 5 : Division des données en ensembles d'entraînement et de test ---")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SET_SIZE, random_state=RANDOM_STATE, stratify=y  
)
print(f"Taille du jeu d'entraînement : {X_train.shape}, Taille du jeu de test : {X_test.shape}")

# =============================================================================
#                 MISE EN PLACE DU PRÉTRAITEMENT AUTOMATIQUE
# =============================================================================
print("\n--- Étape 6 : Définition du pipeline de prétraitement des données ---")
numerical_features_in_train = X_train.select_dtypes(include=np.number).columns.tolist()
categorical_features_in_train = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
print(f"Colonnes numériques qui seront prétraitées : {numerical_features_in_train}")
print(f"Colonnes catégorielles qui seront prétraitées : {categorical_features_in_train}")

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
print("Pipeline de prétraitement configuré.")

# =============================================================================
#           CRÉATION, ENTRAÎNEMENT ET ÉVALUATION DES MODÈLES
# =============================================================================
model_accuracies = {}

print("\n" + "="*40 + "\n MODÈLE 1: RANDOM FOREST \n" + "="*40)
rf_pipeline_model = ImblearnPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=RANDOM_STATE)), 
    ('classifier', RandomForestClassifier(n_estimators=N_ESTIMATORS_RF, random_state=RANDOM_STATE, n_jobs=-1))
])
print("Entraînement du modèle Random Forest")
rf_pipeline_model.fit(X_train, y_train)
print("Modèle Random Forest entraîné !")
y_pred_rf = rf_pipeline_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
model_accuracies['Random Forest'] = accuracy_rf
print(f"\n Évaluation du Random Forest sur le jeu de test")
print(f"Précision globale (Accuracy) : {accuracy_rf:.4f} ({accuracy_rf*100:.2f}%)")
print("Rapport de Classification détaillé :")
print(classification_report(y_test, y_pred_rf, zero_division=0))
print("\nMatrice de Confusion (Random Forest) :")
try:
    cm_rf = confusion_matrix(y_test, y_pred_rf, labels=rf_pipeline_model.classes_) 
    disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=rf_pipeline_model.classes_)
    fig_rf, ax_rf = plt.subplots(figsize=(7, 6)) 
    disp_rf.plot(cmap='Greens', ax=ax_rf, colorbar=True) 
    plt.title('Matrice de Confusion (Random Forest)')
    plt.show() 
except Exception as e_rf:
    print("  Erreur lors de l'affichage de la matrice de confusion pour Random Forest: {e_rf}")

print("\n" + "="*40 + "\n MODÈLE 2: ARBRE DE DÉCISION \n" + "="*40)
dt_pipeline_model = ImblearnPipeline(steps=[
    ('preprocessor', preprocessor), 
    ('smote', SMOTE(random_state=RANDOM_STATE)), 
    ('classifier', DecisionTreeClassifier(random_state=RANDOM_STATE)) 
])
print("Entraînement du modèle Arbre de Décision.")
dt_pipeline_model.fit(X_train, y_train)
print("Modèle Arbre de Décision entraîné ")
y_pred_dt = dt_pipeline_model.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
model_accuracies['Arbre de Décision'] = accuracy_dt
print(f"\nÉvaluation de l'Arbre de Décision sur le jeu de test")
print(f"Précision globale (Accuracy) : {accuracy_dt:.4f} ({accuracy_dt*100:.2f}%)")
print("Rapport de Classification détaillé:")
print(classification_report(y_test, y_pred_dt, zero_division=0))
print("\nMatrice de Confusion (Arbre de Décision) :")
try:
    cm_dt = confusion_matrix(y_test, y_pred_dt, labels=dt_pipeline_model.classes_) 
    disp_dt = ConfusionMatrixDisplay(confusion_matrix=cm_dt, display_labels=dt_pipeline_model.classes_)
    fig_dt, ax_dt = plt.subplots(figsize=(7, 6)) 
    disp_dt.plot(cmap='Oranges', ax=ax_dt, colorbar=True) 
    plt.title('Matrice de Confusion (Arbre de Décision)')
    plt.show() 
except Exception as e_dt:
    print(f"  Erreur lors de l'affichage de la matrice de confusion pour Arbre de Décision: {e_dt}")

# =============================================================================
#            COMPARAISON ET SAUVEGARDE DES MODÈLES
# =============================================================================
print("\n" + "="*40 + "\n BILAN DES PERFORMANCES ET SAUVEGARDE \n" + "="*40)
print("Accuracies obtenues sur le jeu de test :")
for model_name, acc in model_accuracies.items():
    print(f"- {model_name:<20} : {acc:.4f} ({acc*100:.2f}%)")

print(f"\n--- Sauvegarde du pipeline Random Forest pour l'application ---")
print(f"Le pipeline Random Forest sera sauvegardé sous le nom : {PIPELINE_FILENAME}") 
try:
    joblib.dump(rf_pipeline_model, PIPELINE_FILENAME) 
    print(f"Pipeline Random Forest sauvegardé avec succès dans '{PIPELINE_FILENAME}'.")
except Exception as e:
    print(f"ERREUR de sauvegarde du pipeline RF : {e}")

print(f"\n--- Sauvegarde du pipeline Arbre de Décision (pour référence) ---")
print(f"Le pipeline Arbre de Décision sera sauvegardé sous le nom : {PIPELINE_DT_SAVE_NAME}")
try:
    joblib.dump(dt_pipeline_model, PIPELINE_DT_SAVE_NAME)
    print(f"Pipeline Arbre de Décision sauvegardé avec succès dans '{PIPELINE_DT_SAVE_NAME}'.")
except Exception as e:
    print(f"ERREUR de sauvegarde du pipeline DT : {e}")

print("\nSauvegarde des options pour l'interface (menus déroulants)...")
options_dict = {}
for col in categorical_features_in_train: 
    if col in df.columns: 
        unique_values = df[col].dropna().unique().tolist()
        try:
            options_dict[col] = sorted([str(val) for val in unique_values]) 
        except TypeError: 
            options_dict[col] = [str(val) for val in unique_values]
print(f"Sauvegarde des options dans : {OPTIONS_FILENAME}")
try:
    with open(OPTIONS_FILENAME, 'w', encoding='utf-8') as f:
        json.dump(options_dict, f, ensure_ascii=False, indent=4) 
    print("Options pour les menus sauvegardées.")
except Exception as e:
    print(f"ERREUR de sauvegarde des options : {e}")

print("\n--- Script d'entraînement des deux modèles terminé avec succès ! ---")