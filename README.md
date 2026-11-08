# Prédiction du risque de diabète

Projet de classification visant à identifier les patients à risque de diabète à partir de mesures médicales, avec Python et Scikit-learn.

## Données
Jeu de données Pima Indians Diabetes : 768 patientes, 8 variables médicales et la variable cible `Outcome` (1 = diabétique, 0 = non diabétique). Le jeu est modérément déséquilibré (environ 35 % de diabétiques).

Un point important : dans ce jeu, certaines valeurs à 0 sont en réalité des données manquantes (un glucose ou un IMC de 0 n'est pas physiologiquement possible). Elles ont été remplacées par `NaN`, puis imputées par la médiane à l'intérieur de chaque pipeline. Ce traitement évite une fuite de données entre l'entraînement et le test.

| Variable | Valeurs manquantes (0) |
|---|---|
| Glucose | 5 |
| BloodPressure | 35 |
| SkinThickness | 227 |
| Insulin | 374 |
| BMI | 11 |

## Méthode
1. Chargement et vérification des données
2. Traitement des valeurs manquantes (imputation par la médiane dans un `Pipeline`)
3. Séparation apprentissage / test (80 % / 20 %, stratifiée)
4. Comparaison de quatre modèles : régression logistique, SVM, Random Forest, Gradient Boosting
5. Optimisation des hyperparamètres par GridSearchCV avec validation croisée stratifiée à 5 plis
6. Choix du modèle selon l'AUC, le recall et la precision
7. Ajustement du seuil de décision pour privilégier le recall

## Résultats (jeu de test)

| Modèle | AUC | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| **Random Forest** | **0,822** | 0,753 | 0,618 | **0,778** | 0,689 |
| Gradient Boosting | 0,822 | 0,740 | 0,659 | 0,537 | 0,592 |
| Régression logistique | 0,810 | 0,708 | 0,571 | 0,667 | 0,615 |
| SVM | 0,807 | 0,708 | 0,600 | 0,500 | 0,546 |

Le modèle retenu est le **Random Forest**, le seul à atteindre un recall d'au moins 0,70 avec une precision d'au moins 0,50.

### Effet du seuil de décision

| Seuil | Vrais négatifs | Faux positifs | Faux négatifs | Vrais positifs | Recall | Precision |
|---|---|---|---|---|---|---|
| 0,50 (par défaut) | 74 | 26 | 12 | 42 | 0,778 | 0,618 |
| **0,35 (retenu)** | 61 | 39 | 6 | 48 | **0,889** | 0,552 |

En santé, rater un malade est plus grave qu'une fausse alerte. Le seuil 0,35 détecte 48 malades sur 54, au prix de fausses alertes qui demandent un examen complémentaire.

## Limites
- Le seuil de décision a été choisi sur le jeu de test. Les performances présentées sont donc légèrement optimistes. Une évaluation définitive demanderait de choisir le seuil par validation croisée.
- Le jeu de données est petit (768 patientes) et ne concerne qu'une population spécifique : il ne permet pas de conclure sur une population générale.
- Ce modèle est un outil d'aide au dépistage, il ne remplace pas un diagnostic médical.

## Reproduire le projet
```bash
pip install -r requirements.txt
jupyter notebook diabetes_prediction_Aicha.ipynb
```

## Auteur
Aicha Taazza : Master Data Science et Actuariat, Faculté d'Économie et de Gestion, Settat
