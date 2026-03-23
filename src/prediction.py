Income Prediction Script
Module pour prédire le revenu d'une nouvelle personne
"""

import pandas as pd
import numpy as np
import pickle
from typing import Dict, Tuple, Optional


class IncomePredictor:
    """
    Classe pour charger le modèle et faire des prédictions
    """
    
    def __init__(self, models_dir: str = '../models'):
        """
        Initialise le prédicteur en chargeant les modèles sauvegardés
        
        Args:
            models_dir: Chemin vers le dossier contenant les modèles
        """
        self.models_dir = models_dir
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.threshold = 0.3
        
        self._load_models()
    
    def _load_models(self):
        """Charge tous les objets nécessaires"""
        try:
            # Charger le modèle
            with open(f'{self.models_dir}/logistic_regression_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            
            # Charger le scaler
            with open(f'{self.models_dir}/scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            
            # Charger les colonnes
            with open(f'{self.models_dir}/feature_columns.pkl', 'rb') as f:
                self.feature_columns = pickle.load(f)
            
            # Charger le threshold
            try:
                with open(f'{self.models_dir}/threshold.pkl', 'rb') as f:
                    self.threshold = pickle.load(f)
            except FileNotFoundError:
                self.threshold = 0.3  # Valeur par défaut
            
            print("✅ Modèles chargés avec succès")
            
        except Exception as e:
            print(f"❌ Erreur lors du chargement des modèles : {e}")
            raise
    
    def predict(self,
                age: int,
                workclass: str,
                education: str,
                marital_status: str,
                occupation: str,
                sex: str,
                hours_per_week: int,
                capital_gain: int = 0,
                capital_loss: int = 0,
                race: str = 'White',
                relationship: str = None,
                native_country: str = 'United-States',
                fnlwgt: int = 200000) -> Tuple[int, float, Dict]:
        """
        Prédit le revenu d'une personne
        
        Args:
            age: Âge de la personne
            workclass: Type d'employeur
            education: Niveau d'éducation
            marital_status: Statut marital
            occupation: Profession
            sex: Sexe (Male/Female)
            hours_per_week: Heures travaillées par semaine
            capital_gain: Gains de capital (optionnel)
            capital_loss: Pertes de capital (optionnel)
            race: Race (optionnel)
            relationship: Type de relation (optionnel)
            native_country: Pays d'origine (optionnel)
            fnlwgt: Poids final (optionnel)
        
        Returns:
            prediction: 0 (<=50K) ou 1 (>50K)
            probability: Probabilité de >50K
            details: Dictionnaire avec les détails
        """
        
        # Mapping education → education_num
        education_mapping = {
            'Preschool': 1, '1st-4th': 2, '5th-6th': 3, '7th-8th': 4,
            '9th': 5, '10th': 6, '11th': 7, '12th': 8,
            'HS-grad': 9, 'Some-college': 10, 'Assoc-voc': 11, 'Assoc-acdm': 12,
            'Bachelors': 13, 'Masters': 14, 'Prof-school': 15, 'Doctorate': 16
        }
        education_num = education_mapping.get(education, 13)
        
        # Définir relationship si non fourni
        if relationship is None:
            if marital_status == 'Married-civ-spouse':
                relationship = 'Husband' if sex == 'Male' else 'Wife'
            else:
                relationship = 'Not-in-family'
        
        # Créer le profil
        profil = {
            'age': age,
            'workclass': workclass,
            'fnlwgt': fnlwgt,
            'education': education,
            'education_num': education_num,
            'marital_status': marital_status,
            'occupation': occupation,
            'relationship': relationship,
            'race': race,
            'sex': sex,
            'capital_gain': capital_gain,
            'capital_loss': capital_loss,
            'hours_per_week': hours_per_week,
            'native_country': native_country
        }
        
        # Convertir en DataFrame
        df = pd.DataFrame([profil])
        
        # OneHot encoder (comme à l'entraînement)
        nominal_cols = ['workclass', 'marital_status', 'occupation', 
                        'relationship', 'race', 'sex', 'native_country']
        df_encoded = pd.get_dummies(df, columns=nominal_cols, drop_first=True)
        
        # Aligner les colonnes
        for col in self.feature_columns:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        df_encoded = df_encoded[self.feature_columns]
        
        # Normaliser
        df_scaled = self.scaler.transform(df_encoded)
        
        # Prédire
        proba = self.model.predict_proba(df_scaled)[0, 1]
        prediction = 1 if proba >= self.threshold else 0
        
        # Détails
        details = {
            'profil': profil,
            'probability_<=50K': 1 - proba,
            'probability_>50K': proba,
            'threshold': self.threshold,
            'prediction': '>50K' if prediction == 1 else '<=50K',
            'confidence': max(proba, 1 - proba)
        }
        
        return prediction, proba, details


def main():
    """
    Exemple d'utilisation
    """
    print("=" * 80)
    print("INCOME PREDICTION SYSTEM")
    print("=" * 80)
    
    # Initialiser le prédicteur
    predictor = IncomePredictor()
    
    # Exemple 1 : Jeune diplômé
    print("\n📊 EXEMPLE 1 : Jeune diplômé")
    print("-" * 80)
    
    prediction, proba, details = predictor.predict(
        age=25,
        workclass='Private',
        education='Bachelors',
        marital_status='Never-married',
        occupation='Tech-support',
        sex='Male',
        hours_per_week=40
    )
    
    print(f"Probabilité >50K : {proba:.2%}")
    print(f"Prédiction : {details['prediction']}")
    print(f"Confiance : {details['confidence']:.2%}")
    
    # Exemple 2 : Cadre expérimenté
    print("\n📊 EXEMPLE 2 : Cadre expérimenté")
    print("-" * 80)
    
    prediction, proba, details = predictor.predict(
        age=45,
        workclass='Private',
        education='Masters',
        marital_status='Married-civ-spouse',
        occupation='Exec-managerial',
        sex='Male',
        hours_per_week=50,
        capital_gain=5000
    )
    
    print(f"Probabilité >50K : {proba:.2%}")
    print(f"Prédiction : {details['prediction']}")
    print(f"Confiance : {details['confidence']:.2%}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
