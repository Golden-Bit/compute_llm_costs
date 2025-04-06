def calculate_history_tokens(max_pairs: int,
                             avg_tokens_per_message: int) -> float:
    """
    Calcola il numero totale di token dovuti alla 'history' di una chat,
    sulla base del numero massimo di coppie (utente+AI) e della lunghezza
    media (in token) per messaggio.

    Parametri
    ----------
    max_pairs : int
        Numero massimo di coppie utente+AI nello storico.
    avg_tokens_per_message : int
        Numero medio di token per ciascun messaggio.

    Ritorna
    ----------
    float
        Numero totale di token della history, calcolato come:
        2 * max_pairs * avg_tokens_per_message
        (2 messaggi per coppia).
    """
    return 2.0 * max_pairs * float(avg_tokens_per_message)


def build_history_text(max_pairs: int,
                      avg_tokens_per_message: int,
                      user_input_text: str) -> str:
    """
    Costruisce (in modo simulato) un testo di 'history' a partire da un numero massimo
    di coppie di messaggi e da una lunghezza media (in token) di ciascun messaggio.
    Aggiunge in fondo il messaggio di input corrente dell'utente.

    Parametri:
    -----------
    max_pairs : int
        Numero massimo di coppie (utente+AI) da simulare nello storico.
    avg_tokens_per_message : int
        Numero medio di token per ogni messaggio nello storico.
    user_input_text : str
        Testo di input corrente dell'utente.

    Ritorna:
    -----------
    str
        Una stringa che simula l'intero contenuto di 'history' più il messaggio utente finale.
    """

    lines = []
    for i in range(max_pairs):
        # Messaggio utente fittizio (placeholder)
        user_msg = f"[USER_{i+1}] " + ("x " * avg_tokens_per_message)
        # Messaggio AI fittizio (placeholder)
        bot_msg = f"[BOT_{i+1}] " + ("y " * avg_tokens_per_message)

        lines.append(user_msg)
        lines.append(bot_msg)

    # Aggiunge il messaggio di input effettivo alla fine (messaggio "corrente")
    final_user_input = "[USER_INPUT] " + user_input_text

    lines.append(final_user_input)

    # Unisce tutto con newline
    return "\n".join(lines)


def cost_of_message(
    # ---- parametri per dimensionare T_in ----
    max_pairs: int,
    avg_tokens_per_message: int,
    user_tokens: float,
    n_kbox: float,
    r_per_kbox: float,
    chunk_size: float,
    # ---- parametri per l'output ----
    out_tokens: float,
    # ---- costi per token in/out ----
    p_in: float,
    p_out: float,
    # ---- costi retrieval e store ----
    c_retrieval: float,
    c_store: float
) -> float:
    """
    Calcola il costo di un singolo messaggio (turno domanda+risposta) secondo
    il modello di costo esteso, dipendente da tutti i parametri.

    Parametri:
    -----------
    max_pairs : int
        Numero massimo di coppie nello storico.
    avg_tokens_per_message : int
        Numero medio di token per ogni messaggio nello storico.
        Usato insieme a 'max_pairs' per calcolare i token totali di history.
    user_tokens : float
        Numero di token relativi al messaggio utente corrente (input dell'utente).
    n_kbox : float
        Numero medio di KBox interrogate (es. 1.5).
    r_per_kbox : float
        Numero di chunk (risultati) restituiti per ogni KBox (es. 3).
    chunk_size : float
        Lunghezza media (in token) di ciascun chunk di retrieval (es. 300).
    out_tokens : float
        Numero di token di output, ossia la lunghezza media in token della risposta AI.
    p_in : float
        Costo (in dollari) per 1k token di input (prompt).
    p_out : float
        Costo (in dollari) per 1k token di output (risposta).
    c_retrieval : float
        Costo fisso per retrieval (embedding query, query vettoriale, lettura chunk).
    c_store : float
        Costo fisso per salvare conversazione (scritture DB, eventuale embedding conversazione).

    Ritorna:
    -----------
    float
        Costo totale del singolo messaggio in dollari, secondo la formula:

        T_in = T_history + user_tokens + (n_kbox * r_per_kbox * chunk_size)
        C_msg = ( (T_in / 1000)*p_in + (out_tokens / 1000)*p_out )
                + c_retrieval + c_store
    """

    # Calcola i token dovuti allo storico, usando la funzione dedicata:
    history_tokens = calculate_history_tokens(max_pairs, avg_tokens_per_message)

    # Calcolo dei token totali di input
    t_in = history_tokens + user_tokens + (n_kbox * r_per_kbox * chunk_size)

    # Costo LLM = (token_in/1000 * p_in) + (token_out/1000 * p_out)
    cost_llm = (t_in / 1000.0) * p_in + (out_tokens / 1000.0) * p_out

    # Somma costi retrieval + store
    cost_total = cost_llm + c_retrieval + c_store
    return cost_total

if __name__ == "__main__":
    # Esempio di costruzione di una history di prova (parametri fissi)
    simulated_history = build_history_text(
        max_pairs=25,
        avg_tokens_per_message=100,
        user_input_text="Questo è il messaggio utente corrente"
    )
    print("Esempio di history simulata (prime 500 caratteri):\n")
    print(simulated_history[:500] + "...\n")

    # Parametri fissi o semivariabili
    avg_tokens_per_msg = 100
    user_tokens_ex = 50
    chunk_size_ex = 300
    out_tokens_ex = 300

    # Costi retrieval/store fissati
    c_retr_ex = 1e-5
    c_store_ex = 1e-5

    # Definizione modelli e rispettivi costi in/out
    models = {
        "GPT-4o":   {"p_in": 0.005,   "p_out": 0.015},
        "GPT-4oMini": {"p_in": 0.00015, "p_out": 0.00060}
    }

    # Valori da combinare
    max_pairs_list = [10, 20, 30]
    n_kbox_list    = [1, 2, 3]
    r_per_kbox_list = [5, 10, 15]

    # Cicli su tutte le combinazioni
    for max_pairs_ex in max_pairs_list:
        for n_kbox_ex in n_kbox_list:
            for r_per_kbox_ex in r_per_kbox_list:
                for model_name, model_costs in models.items():
                    p_in_ex  = model_costs["p_in"]
                    p_out_ex = model_costs["p_out"]

                    # Calcolo costo
                    cost_example = cost_of_message(
                        max_pairs=max_pairs_ex,
                        avg_tokens_per_message=avg_tokens_per_msg,
                        user_tokens=user_tokens_ex,
                        n_kbox=n_kbox_ex,
                        r_per_kbox=r_per_kbox_ex,
                        chunk_size=chunk_size_ex,
                        out_tokens=out_tokens_ex,
                        p_in=p_in_ex,
                        p_out=p_out_ex,
                        c_retrieval=c_retr_ex,
                        c_store=c_store_ex
                    )

                    # Stampa risultati
                    print(f"Risultato cost_of_message -> max_pairs={max_pairs_ex}, "
                          f"n_kbox={n_kbox_ex}, r_per_kbox={r_per_kbox_ex}, "
                          f"model={model_name} => costo={cost_example:.6f} $")
    print("\n*** Fine combinazioni ***\n")
