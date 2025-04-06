import math

def tokens_for_resolution(width: int, height: int) -> float:
    """
    Stima il numero di token equivalenti per un'immagine/frame
    data la sua risoluzione (width x height in pixel).

    In base ai dati di riferimento:
      - 512x512 px ~ 255 token
    possiamo approssimare con un fattore di proporzionalità.

    Parametri:
    -----------
    width, height : int
        Dimensioni in pixel dell'immagine/frame.

    Ritorna:
    -----------
    float
        Numero stimato di token equivalenti per rappresentare l'immagine.
    """
    # Riferimento: 512x512 = 262144 px => 255 token => ratio ~ 1 token ogni ~1028 px
    # (Si può fare una stima lineare)
    px_count = width * height
    ratio_tokens_per_px = 255.0 / (512.0 * 512.0)  # ~ 255 / 262144
    return px_count * ratio_tokens_per_px


def cost_upload_pdf(
    # Numero di pagine
    n_pages: int,
    # Costo per pagina (0.001 -> Fast, 0.01 -> Hi-Res)
    c_page: float,
    # Numero totale di token estratti dal PDF
    t_total: float,
    # Chunk size in token (es. 500)
    barT_chunk: float,
    # Costo embedding per 1k token (es. 0.00002)
    c_embed: float,
    # Costo scrittura di 1 chunk su DB+Vector (es. 7.5e-6)
    c_db_chunk: float
) -> float:
    """
    Calcola il costo di caricamento e indicizzazione di un documento PDF/testuale
    utilizzando Unstructured (pipeline Fast o Hi-Res), generazione embedding e scrittura DB.

    Parametri:
    ----------
    n_pages : int
        Numero di pagine del documento.
    c_page : float
        Costo per pagina, e.g.:
         - 0.001 (Fast Pipeline)
         - 0.01  (Hi-Res Pipeline)
    t_total : float
        Numero totale di token estratti dal PDF (es. 5000 per 10 pagine).
    barT_chunk : float
        Dimensione in token di un chunk (es. 500).
    c_embed : float
        Costo per generare embedding di 1k token (es. 0.00002).
    c_db_chunk : float
        Costo scrittura di 1 chunk + embedding su DB (es. 7.5e-6).

    Ritorna:
    -----------
    float
        Costo totale (in dollari) del caricamento (processing + embedding + DB).
    """
    # 1) Costo di processing (Unstructured pipeline)
    c_processing = n_pages * c_page

    # 2) Numero chunk
    n_chunk = math.ceil(t_total / barT_chunk)

    # 3) Costo embedding
    c_embed_chunk = (barT_chunk / 1000.0) * c_embed
    c_embedding = n_chunk * c_embed_chunk

    # 4) Costo DB
    c_db = n_chunk * c_db_chunk

    return c_processing + c_embedding + c_db


def cost_upload_image(
    # Risoluzione immagine
    width: int,
    height: int,
    # Token output di didascalia
    t_descr: float,
    # Costo input LLM (es. 0.005 GPT-4o, 0.00015 GPT-4o Mini) [$/1k token]
    p_in: float,
    # Costo output LLM (es. 0.015 GPT-4o, 0.00060 GPT-4o Mini) [$/1k token]
    p_out: float,
    # Costo embedding per 1k token
    c_embed: float,
    # Costo DB (scrittura chunk+embedding)
    c_db_chunk: float
) -> float:
    """
    Calcola il costo di caricamento e indicizzazione di un'immagine,
    includendo la creazione di una caption testuale mediante GPT-4o/Mini,
    l'embedding della didascalia e la scrittura su DB.

    Parametri:
    -----------
    width, height : int
        Dimensioni dell'immagine (pixel).
    t_descr : float
        Numero di token medi della descrizione (es. 100).
    p_in : float
        Costo (in $/1k token) per l'input (prompt) LLM.
    p_out : float
        Costo (in $/1k token) per l'output (risposta) LLM.
    c_embed : float
        Costo embedding (in $) per 1k token.
    c_db_chunk : float
        Costo scrittura su DB di un chunk + embedding (es. ~7.5e-6).

    Ritorna:
    -----------
    float
        Costo totale (in dollari) per caricare l'immagine,
        generare didascalia, embedding e salvare nel DB.
    """
    # 1) Calcolo dei token equivalenti dell'immagine
    t_img = tokens_for_resolution(width, height)

    # 2) Costo LLM caption (input + output)
    cost_llm_caption = (t_img / 1000.0) * p_in + (t_descr / 1000.0) * p_out

    # 3) Costo embedding della descrizione
    cost_embedding = (t_descr / 1000.0) * c_embed

    # 4) Costo DB
    cost_db = c_db_chunk  # un singolo chunk con embedding

    return cost_llm_caption + cost_embedding + cost_db


def cost_upload_video(
    # Durata e sampling
    duration_sec: float,
    sampling_sec: float,
    # Risoluzione frame
    width: int,
    height: int,
    # Token output (descr) per 1 frame
    t_descr: float,
    # Costo LLM input e output
    p_in: float,
    p_out: float,
    # Costo embedding per 1k token
    c_embed: float,
    # Costo DB (scrittura chunk+embedding)
    c_db_chunk: float,
    # Numero token totali (unificati) per l'embedding finale
    t_descr_total: float
) -> float:
    """
    Calcola il costo di caricamento di un video, considerando l'estrazione
    di frame ogni 'sampling_sec' secondi, la generazione di una caption
    per ciascun frame tramite GPT-4o/Mini, poi un embedding finale (usando
    t_descr_total come total token finali).

    Parametri:
    -----------
    duration_sec : float
        Durata del video in secondi.
    sampling_sec : float
        Intervallo in secondi per estrarre un frame (es. 10s).
    width, height : int
        Dimensioni in pixel di ciascun frame.
    t_descr : float
        Token output di descrizione per 1 frame (es. 50).
    p_in, p_out : float
        Costi input e output LLM (in $/1k token).
    c_embed : float
        Costo embedding (in $) per 1k token.
    c_db_chunk : float
        Costo scrittura di un chunk+embedding.
    t_descr_total : float
        Numero di token totali di tutte le didascalie combinate.
        (p.es. n_frame * t_descr, se uniamo tutto in un singolo testo)

    Ritorna:
    -----------
    float
        Costo totale per processare il video (LLM su each frame) + embedding + DB.
    """
    # 1) Calcolo n_frame
    n_frame = math.ceil(duration_sec / sampling_sec)

    # 2) Token equivalenti di un frame
    t_img = tokens_for_resolution(width, height)

    # 3) Costo LLM su each frame
    #    (input t_img e output t_descr)
    cost_llm_one_frame = (t_img / 1000.0) * p_in + (t_descr / 1000.0) * p_out
    cost_llm_all_frames = cost_llm_one_frame * n_frame

    # 4) Costo embedding finale
    cost_embedding_final = (t_descr_total / 1000.0) * c_embed

    # 5) Costo DB (1 chunk unificato)
    cost_db = c_db_chunk

    return cost_llm_all_frames + cost_embedding_final + cost_db



def run_experiments_pdf_image_video():
    """
    Esegue la chiamata a cost_upload_pdf, cost_upload_image, cost_upload_video
    su tutte le combinazioni richieste, senza tralasciare nulla:

    1) PDF (documenti):
       - pipeline in ["HiRes","Fast"] => c_page=0.01 / 0.001
       - n_pages in [5,10,20,50,100]
       - Nessun loop su modello (ingestion doc non dipende dal modello)

    2) Immagini:
       - risoluzioni in [(512,512),(1024,1024),(2048,2048)]
       - modello in ["GPT-4o","GPT-4oMini"]

    3) Video (durata=1 minuto => 60s):
       - risoluzioni in [(512,512),(1024,1024),(2048,2048)]
       - "frame rate" in [1,0.5,0.2,0.1] => n_frames = 60 * rate
       - modello in ["GPT-4o","GPT-4oMini"]

    Stampera' i risultati di costo per ogni combinazione.
    """

    # (A) Parametri comuni
    c_embed_pdf = 0.00002
    c_db_chunk_pdf = 7.5e-6
    barT_chunk_pdf = 500  # chunk di 500 token

    c_embed_img = 0.00002
    c_db_chunk_img = 7.5e-6
    t_descr_img = 100  # 100 token di descrizione

    c_embed_vid = 0.00002
    c_db_chunk_vid = 7.5e-6
    t_descr_frame = 50  # 50 token per frame

    print("==============================================")
    print("ESPERIMENTI DI CARICAMENTO PDF (no modello).")
    print("==============================================\n")

    pipeline_options = ["HiRes", "Fast"]   # => c_page=0.01 o 0.001
    pages_list = [5, 10, 20, 50, 100]

    for pipeline in pipeline_options:
        if pipeline == "HiRes":
            c_page = 0.01
        else:
            c_page = 0.001

        for n_pages in pages_list:
            # ipotesi: ~500 token/pagina
            t_total_doc = 500 * n_pages

            cost_pdf = cost_upload_pdf(
                n_pages=n_pages,
                c_page=c_page,
                t_total=t_total_doc,
                barT_chunk=barT_chunk_pdf,
                c_embed=c_embed_pdf,
                c_db_chunk=c_db_chunk_pdf
            )
            print(f"[PDF] pipeline={pipeline}, pages={n_pages} "
                  f"-> cost={cost_pdf:.6f} $ (c_page={c_page}, t_total={t_total_doc}, chunk=500)")

    print("\n\n==============================================")
    print("ESPERIMENTI DI CARICAMENTO IMMAGINI (con modelli).")
    print("==============================================\n")

    resolutions_img = [(512,512), (1024,1024), (2048,2048)]
    models = {
        "GPT-4o": {"p_in": 0.005, "p_out": 0.015},
        "GPT-4oMini": {"p_in": 0.00015, "p_out": 0.00060}
    }

    for (w,h) in resolutions_img:
        for model_name, model_costs in models.items():
            cost_img = cost_upload_image(
                width=w,
                height=h,
                t_descr=t_descr_img,
                p_in=model_costs["p_in"],
                p_out=model_costs["p_out"],
                c_embed=c_embed_img,
                c_db_chunk=c_db_chunk_img
            )
            print(f"[IMG] resolution={w}x{h}, model={model_name} -> cost={cost_img:.6f} $ "
                  f"(descr={t_descr_img} token)")

    print("\n\n==============================================")
    print("ESPERIMENTI DI CARICAMENTO VIDEO (1 minuto, con modelli).")
    print("==============================================\n")

    # 1 minuto => 60 secondi
    # "frame rate" in [1, 0.5, 0.2, 0.1]
    # => n_frames = 60 * rate
    # => sampling_sec = 60 / n_frames
    # => t_descr_total = n_frames * t_descr_frame

    resolutions_vid = [(512,512), (1024,1024), (2048,2048)]
    rates = [1.0, 0.5, 0.2, 0.1]  # fps

    for (w,h) in resolutions_vid:
        for rate in rates:
            n_frames = int( round(60.0 * rate) )
            if n_frames < 1:
                n_frames = 1  # per sicurezza

            sampling_sec = 60.0 / n_frames
            t_descr_total = n_frames * t_descr_frame

            for model_name, model_costs in models.items():
                cost_vid = cost_upload_video(
                    duration_sec=60.0,  # 1 min
                    sampling_sec=sampling_sec,
                    width=w,
                    height=h,
                    t_descr=t_descr_frame,
                    p_in=model_costs["p_in"],
                    p_out=model_costs["p_out"],
                    c_embed=c_embed_vid,
                    c_db_chunk=c_db_chunk_vid,
                    t_descr_total=t_descr_total
                )
                print(f"[VIDEO] duration=1min, rate={rate}fps => frames={n_frames}, resolution={w}x{h}, model={model_name} "
                      f"-> cost={cost_vid:.6f} $ "
                      f"(frame_descr={t_descr_frame}, total_descr={t_descr_total}, sampling={sampling_sec:.3f}s)")

if __name__ == "__main__":
    run_experiments_pdf_image_video()
    print("\n*** Fine di tutte le combinazioni PDF/immagine/video ***\n")