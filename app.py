import streamlit as st
import pandas as pd
import joblib

# Charger le modèle, le scaler et les colonnes
model = joblib.load('C:/Users/Original Shop/OneDrive/Desktop/ENSIAS/Stage_1A/Prediction_turnover/employee_churn_model.pkl')
scaler = joblib.load('C:/Users/Original Shop/OneDrive/Desktop/ENSIAS/Stage_1A/Prediction_turnover/scaler.pkl')
feature_columns = joblib.load('C:/Users/Original Shop/OneDrive/Desktop/ENSIAS/Stage_1A/Prediction_turnover/feature_columns.pkl')

# Titre de l'application
st.title("Prédiction de départ des employés")
st.write("Cette application utilise un modèle de Machine Learning pour prédire si un employé quittera ou non l'entreprise.")

# Fonction pour faire des prédictions
def predict_churn(input_data, feature_columns):
    # Créer un dictionnaire avec toutes les colonnes initialisées à 0
    data_dict = {col: 0 for col in feature_columns}
    
    # Mettre à jour les valeurs numériques
    data_dict.update({
        'review': input_data['review'],
        'projects': input_data['projects'],
        'avg_hrs_month': input_data['avg_hrs_month'],
        'satisfaction': input_data['satisfaction'],
        'bonus': input_data['bonus'],
        'promoted_1': input_data['promoted_1']
    })
    
    # Mettre à jour le département
    dept_col = f"department_{input_data['department']}"
    if dept_col in feature_columns:
        data_dict[dept_col] = 1
        
    # Mettre à jour le niveau de salaire
    salary_col = f"salary_{input_data['salary_level']}"
    if salary_col in feature_columns:
        data_dict[salary_col] = 1
    
    # Créer le DataFrame avec les colonnes dans le bon ordre
    input_df = pd.DataFrame([data_dict], columns=feature_columns)
    
    # Standardiser les données
    input_scaled = scaler.transform(input_df)
    
    # Faire une prédiction
    prediction = model.predict(input_scaled)
    return prediction[0]

# Formulaire pour entrer les données
st.sidebar.header("Entrer les données de l'employé")

# Collecter les entrées utilisateur dans un dictionnaire
input_data = {
    'review': st.sidebar.number_input("Valeur de review", min_value=0.0, max_value=1.0, value=0.5, step=0.01),
    'projects': st.sidebar.number_input("Valeur de projects", min_value=0, max_value=10, value=3, step=1),
    'avg_hrs_month': st.sidebar.number_input("Valeur de avg_hrs_month", min_value=0, max_value=300, value=150, step=1),
    'satisfaction': st.sidebar.number_input("Valeur de satisfaction", min_value=0.0, max_value=1.0, value=0.5, step=0.01),
    'bonus': st.sidebar.number_input("Valeur de bonus", min_value=0, max_value=1, value=0, step=1),
    'department': st.sidebar.selectbox(
        "Département",
        ['operations', 'support', 'logistics', 'sales', 'IT', 'admin', 'engineering', 'marketing', 'finance', 'retail']
    ),
    'salary_level': st.sidebar.selectbox(
        "Niveau de salaire",
        ["low", "medium", "high"]
    ),
    'promoted_1': st.sidebar.selectbox(
        "Promoted (0=Non, 1=Oui)",
        [0, 1]
    )
}

# Bouton pour faire une prédiction
if st.sidebar.button("Prédire"):
    try:
        prediction = predict_churn(input_data, feature_columns)
        if prediction == 1:
            st.error("L'employé est susceptible de quitter l'entreprise.")
        else:
            st.success("L'employé est susceptible de rester dans l'entreprise.")
    except Exception as e:
        st.error(f"Une erreur est survenue lors de la prédiction : {str(e)}")

# Afficher les importances des caractéristiques
if hasattr(model, 'feature_importances_'):
    importances = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    # Supprimer les colonnes "department" si nécessaire
    importances = importances[~importances['Feature'].str.startswith('department')]

    # Afficher les 10 caractéristiques les plus importantes dans Streamlit
    st.header("Top 10 des caractéristiques les plus importantes")
    col1, col2 = st.columns(2)

    with col1:
        st.bar_chart(importances.head(10).set_index('Feature'))

    with col2:
        st.write("Les pourcentages affichés dans le diagramme indiquent l'influence relative de chaque caractéristique dans le modèle.")
        st.write("Ces informations peuvent aider les RH et les managers à identifier les facteurs clés de départ des employés.")