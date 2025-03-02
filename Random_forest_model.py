# # Importations nécessaires
# import pandas as pd
# from imblearn.over_sampling import SMOTE
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# import joblib
# import warnings

# warnings.filterwarnings("ignore", category=FutureWarning)

# # Lire les données
# file_path = r"./employee_churn_data.csv"
# df = pd.read_csv(file_path)

# # Traitement des données
# df.drop('tenure', axis=1, inplace=True)
# df_encoded = pd.get_dummies(df, columns=["department", "promoted", "salary", "left"], drop_first=True)

# # Attributs et résultats
# X = df_encoded.iloc[:, :-1]
# Y = df_encoded.iloc[:, -1]

# # Appliquer SMOTE pour équilibrer les classes
# smote = SMOTE(sampling_strategy='auto', random_state=42)
# X_resampled, Y_resampled = smote.fit_resample(X, Y)

# # Split des données
# X_train, X_test, Y_train, Y_test = train_test_split(X_resampled, Y_resampled, test_size=0.3, random_state=42)

# # Standardiser les données
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # Entraînement du modèle avec les paramètres optimaux
# rf = RandomForestClassifier(
#     n_estimators=100,
#     max_depth=20,
#     min_samples_split=5,
#     min_samples_leaf=1,
#     bootstrap=False,
#     random_state=42
# )

# rf.fit(X_train_scaled, Y_train)

# # Importance des caractéristiques
# feature_importances = pd.DataFrame({
#     'Feature': X.columns,
#     'Importance': rf.feature_importances_
# }).sort_values(by='Importance', ascending=False)


# # Afficher le DataFrame des importances
# print(feature_importances)

# # Évaluation du modèle
# Y_pred = rf.predict(X_test_scaled)
# print("\nConfusion Matrix:")
# print(confusion_matrix(Y_test, Y_pred))
# print("\nClassification Report:")
# print(classification_report(Y_test, Y_pred))
# print("\nAccuracy Score:", accuracy_score(Y_test, Y_pred))

# # Sauvegarde du modèle
# joblib.dump(rf, 'employee_churn_model.pkl')
# print("Modèle Random Forest sauvegardé sous le nom 'employee_churn_model.pkl'.")

# # Sauvegarde du scaler
# scaler = StandardScaler()
# scaler.fit(X_train)
# joblib.dump(scaler, 'scaler.pkl')
# print("Scaler sauvegardé sous le nom 'scaler.pkl'.")



# Importations nécessaires
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import warnings
import pickle

warnings.filterwarnings("ignore", category=FutureWarning)

# Lire les données
file_path = r"./employee_churn_data.csv"
df = pd.read_csv(file_path)

# Traitement des données
df.drop('tenure', axis=1, inplace=True)
df_encoded = pd.get_dummies(df, columns=["department", "promoted", "salary", "left"], drop_first=True)

# Attributs et résultats
X = df_encoded.iloc[:, :-1]
Y = df_encoded.iloc[:, -1]

# Appliquer SMOTE pour équilibrer les classes
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, Y_resampled = smote.fit_resample(X, Y)

# Split des données
X_train, X_test, Y_train, Y_test = train_test_split(X_resampled, Y_resampled, test_size=0.3, random_state=42)

# Standardiser les données
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entraînement du modèle avec les paramètres optimaux
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=1,
    bootstrap=False,
    random_state=42
)

rf.fit(X_train_scaled, Y_train)

# Importance des caractéristiques
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Afficher le DataFrame des importances
print(feature_importances)

# Évaluation du modèle
Y_pred = rf.predict(X_test_scaled)
print("\nConfusion Matrix:")
print(confusion_matrix(Y_test, Y_pred))
print("\nClassification Report:")
print(classification_report(Y_test, Y_pred))
print("\nAccuracy Score:", accuracy_score(Y_test, Y_pred))

# Sauvegarde du modèle
joblib.dump(rf, 'employee_churn_model.pkl')
print("Modèle Random Forest sauvegardé sous le nom 'employee_churn_model.pkl'.")

# Sauvegarde du scaler
joblib.dump(scaler, 'scaler.pkl')
print("Scaler sauvegardé sous le nom 'scaler.pkl'.")

# **Sauvegarde des noms des colonnes**
with open('feature_columns.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)
print("Noms des colonnes sauvegardés sous le nom 'feature_columns.pkl'.")

