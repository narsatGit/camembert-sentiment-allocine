import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from datasets import load_dataset

from preprocessing import (
    recup_review_label,
    del_punct,
    tokeniser,
    id_asso,
    bag_of_words
)
from model import ReseauTest


def main():
    # Fixer la graine
    torch.manual_seed(42)
    
    r"""Exécute l'ensemble de la chaîne d'entraînement et d'évaluation du modèle."""
    # 1. Chargement et préparation du Dataset Allociné
    print("Chargement du dataset Allociné...")
    ds = load_dataset("tblard/allocine")

    ds_shuffled_train = ds["train"].shuffle(seed=42)
    ds_shuffled_test = ds["test"].shuffle(seed=42)

    train_subset = ds_shuffled_train.select(range(2000))
    test_subset = ds_shuffled_test.select(range(500))

    # 2. Prétraitement et création du vocabulaire (Train)
    reviews_train, labels_train = recup_review_label(train_subset)
    critiques_propres_train = del_punct(reviews_train)
    corpus_tokens_train = tokeniser(critiques_propres_train)
    id_vocabulaire = id_asso(corpus_tokens_train)

    # 3. Prétraitement (Test)
    reviews_test, labels_test = recup_review_label(test_subset)
    critiques_propres_test = del_punct(reviews_test)
    corpus_tokens_test = tokeniser(critiques_propres_test)

    # 4. Vectorisation Bag-of-Words
    print("Vectorisation des données...")
    X_train = bag_of_words(corpus_tokens_train, id_vocabulaire)
    y_train = torch.tensor(labels_train, dtype=torch.float32).unsqueeze(1)

    X_test = bag_of_words(corpus_tokens_test, id_vocabulaire)
    y_test = torch.tensor(labels_test, dtype=torch.float32).unsqueeze(1)

    # 5. Création du DataLoader
    all_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    cutout_dataset = torch.utils.data.DataLoader(all_dataset, batch_size=32, shuffle=True)

    # 6. Initialisation du modèle avec la taille exacte du vocabulaire
    input_dim = len(id_vocabulaire)
    modele = ReseauTest(input_dim=input_dim)
    loss_fn = nn.BCELoss()
    optimiseur = torch.optim.Adam(modele.parameters(), lr=0.01)

    # 7. Boucle d'entraînement
    print("Début de l'entraînement...")
    historique_loss = []
    for epoch in range(10):
        nb_loss = len(cutout_dataset)
        total_loss = 0

        for X_batch, y_batch in cutout_dataset:
            predictions = modele(X_batch)
            loss = loss_fn(predictions, y_batch)

            optimiseur.zero_grad()
            loss.backward()
            optimiseur.step()

            total_loss += loss.item()

        moy_loss = total_loss / nb_loss
        historique_loss.append(moy_loss)
        print(f"Époque {epoch + 1}, Loss : {moy_loss:.4f}")

    # 8. Sauvegarde de la courbe de perte
    os.makedirs("resultats", exist_ok=True)
    plt.plot(historique_loss)
    plt.xlabel("Époque")
    plt.ylabel("Loss")
    plt.title("Évolution de la loss pendant l'entraînement")
    plt.savefig("resultats/courbe_loss.png")
    plt.close()
    print("Courbe de loss sauvegardée dans resultats/courbe_loss.png")

    # 9. Évaluation sur l'ensemble de test
    modele.eval()
    with torch.no_grad():
        predictions_test = modele(X_test)
        classes_predites = (predictions_test > 0.5).float()
        accuracy = (classes_predites == y_test).float().mean()
        print(f"Précision sur le jeu de test : {accuracy.item() * 100:.2f}%")


if __name__ == "__main__":
    main()