import re
import torch


def recup_review_label(dataset):
    r"""Extrait les textes de critiques et leurs étiquettes associées à partir d'un objet dataset.

    Args:
        dataset: Sous-ensemble du dataset Allociné.

    Returns:
        tuple: (reviews, labels) où reviews est une liste de chaînes et labels une liste d'entiers.
    """
    reviews = []
    labels = []
    for sample in dataset:
        review = sample['review']
        label = sample['label']
        reviews.append(review)
        labels.append(label)
    return reviews, labels


def del_punct(critiques):
    r"""Supprime la ponctuation inutile d'une liste de critiques à l'aide d'expressions régulières.

    Args:
        critiques (list): Liste de chaînes de caractères représentant les critiques.

    Returns:
        list: Liste de critiques nettoyées.
    """
    critiques_propres = []
    for critique in critiques:
        critique_propre = re.sub(r"[^\w\s'-]", "", critique)
        critiques_propres.append(critique_propre)
    return critiques_propres


def tokeniser(text):
    r"""Passe le texte en minuscules et le découpe en liste de mots par espaces.

    Args:
        text (list): Liste de phrases/critiques nettoyées.

    Returns:
        list: Liste de listes de tokens (mots).
    """
    corpus_tokens = [critique.lower().split() for critique in text]
    return corpus_tokens


def id_asso(corpus_tokens):
    r"""Construit le dictionnaire de vocabulaire en associant un identifiant unique à chaque mot.

    Args:
        corpus_tokens (list): Liste de listes de tokens d'entraînement.

    Returns:
        dict: Dictionnaire de correspondance {mot: id_entier}.
    """
    tokens_propres = []
    for tokens in corpus_tokens:
        for token in tokens:
            if token.strip() != "":
                tokens_propres.append(token)

    vocab_id = {}
    tokens_uniques = dict.fromkeys(tokens_propres)
    for i, token in enumerate(tokens_uniques):
        vocab_id[token] = i
    return vocab_id


def bag_of_words(corpus_tokens, id_vocabulaire):
    r"""Vectorise un corpus de tokens sous la forme d'un tenseur Bag-of-Words.

    Args:
        corpus_tokens (list): Liste de listes de tokens.
        id_vocabulaire (dict): Dictionnaire de vocabulaire {mot: id}.

    Returns:
        torch.Tensor: Tenseur PyTorch de forme (nb_échantillons, taille_vocabulaire).
    """
    vecteurs = []
    for tokens in corpus_tokens:
        vecteur_courant = torch.zeros(len(id_vocabulaire))
        for token in tokens:
            if token in id_vocabulaire:
                indice = id_vocabulaire[token]
                vecteur_courant[indice] += 1
        vecteurs.append(vecteur_courant)
    return torch.stack(vecteurs)