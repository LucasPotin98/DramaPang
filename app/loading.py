# loader.py
import streamlit as st
from pang.pang import (
    pang_load_and_represent,
    load_titles,
    load_model,
)  # adapte selon tes chemins


def load_dracor_data():
    if "dracor_data" not in st.session_state:
        FILE_GRAPHS = "data/graphs/dracor_graphs.txt"
        FILE_PATTERNS = "data/graphs/dracor_patterns.txt"
        FILE_LABELS = "data/graphs/dracor_labels.txt"
        FILE_TITLES = "data/graphs/dracor_titles.txt"
        FILE_MODEL = "models/pang_model.pkl"

        X_full, Graphes, Patterns, labels, noms = pang_load_and_represent(
            FILE_GRAPHS, FILE_PATTERNS, FILE_LABELS
        )
        titles = load_titles(FILE_TITLES)
        model = load_model(FILE_MODEL)

        st.session_state.dracor_data = (
            X_full,
            Graphes,
            Patterns,
            labels,
            titles,
            noms,
            model,
        )

    return st.session_state.dracor_data
