# 🎭 DramaPang – Et si le genre d’une pièce de théâtre pouvait se lire dans son réseau de personnages ?

> Une application pour explorer les réseaux de personnages dans les pièces françaises, détecter les motifs clés et prédire s’il s’agit d’une **comédie** ou d’une **tragédie**.

---

## Essayez l’application

🟢 Application déployée ici :  
👉 **[DramaPang sur Streamlit →](https://dramapang.streamlit.app/)**  

---

## Objectif du projet

**DramaPang** est un outil interactif pour explorer des graphes de personnages extraits de pièces de théâtre françaises (corpus DraCor), et les **classifier automatiquement** en comédies ou tragédies.  Le cœur du projet repose sur un pipeline basé **sur les motifs discriminants de graphe**. 

---

## Pipeline général

Voici les grandes étapes du processus, du traitement des données à la prédiction du genre :

### 1. Construction des graphes

- Source : corpus [DraCor](https://dracor.org/)
- Un graphe par pièce :
  - **Nœuds** : personnages, labellisés par genre (`MALE`, `FEMALE`, `UNKNOWN`)
  - **Arêtes** : co-présence dans les actes (poids discrétisé)

### 2. Extraction et vectorisation des motifs

- Extraction de **sous-graphes fréquents**
- Représentation des pièces sous forme de vecteurs de motifs
- Scorage selon une mesure sélectionnée :
  - `Sup` : support
  - `AbsSupDif` : différence absolue de support
  - `WRAcc` : précision relative pondérée

### 3. Sélection des motifs discriminants

- Clustering pour réduire la redondance
- Sélection des motifs les plus représentatifs
- Affichage interactif des **top motifs**

### 4. Classification et interprétation

- Modèle : **Arbre de décision**
- Prédiction du genre (`Comédie` ou `Tragédie`)
- Visualisation de :
  - l’arbre de décision
  - le chemin parcouru par l’exemple courant
  - les **4 motifs clés** qui ont conduit à la décision

---

## Données

- **400 pièces** issues de DraCor : 200 comédies, 200 tragédies  
- Chaque pièce est représentée comme un **graphe connexe**

### Structure des graphes

- **Nœuds** : personnages
  - Label lié au `genre` :
    - `MALE` → représenté en bleu
    - `FEMALE` → représenté en rose
    - `UNKNOWN` → représenté en gris

- **Arêtes** : co-présence de deux personnages dans un ou plusieurs actes
  - Pondération discrétisée en trois niveaux :
    - 1 seule co-présence → représentée en **gris**
    - 2 à 5 co-présences → représentée en **noir**
    - plus de 5 co-présences → représentée en **rouge**

---

## Stack technique

- Python : `pandas`, `networkx`, `scikit-learn`, `matplotlib`, `plotly`
- Application : `Streamlit`

---

## À propos du projet

Ce projet a été conçu dans le cadre de ma thèse sur la **détection de motifs discriminants dans les graphes de marchés publics**. Ici, les techniques développées sont réutilisées dans un contexte culturel (théâtre) à des fins pédagogiques et exploratoires.

Le cœur du pipeline repose sur le framework **PANG (Pattern-based Anomaly detection in Graphs)**, développé pour des cas d’usage réels, et adapté ici à un jeu de données open source.

Framework PANG : [github.com/CompNet/PANG](https://github.com/CompNet/PANG)  

---

## 👨‍💻 Auteur

Projet développé par **[Lucas Potin](https://lucaspotin98.github.io/)**  
*Data Scientist – Modélisation & Graphes*
