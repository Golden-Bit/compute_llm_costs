"""
File: script_three_debug.py

Scopo: importare funzioni da `script_one.py` e `script_two.py` e definire una nuova
       funzione `cost_monthly_user_debug(...)` che calcola il costo medio mensile di un utente
       in modo dettagliato, mostrando un debug completo di tutte le voci di costo
       (chat, ingestion di doc/img/video, e storage).

NOVITÀ:
- Possibile fissare la frazione di messaggi scambiati con GPT-4o (es. fraction_4o = 0.2),
  mentre la parte restante (1 - fraction_4o) va su GPT-4o Mini.
- Analoga logica per immagini/video: frazione "fraction_4o_imgvid" che usa GPT-4o,
  e (1 - fraction_4o_imgvid) che usa GPT-4o Mini per la caption.
- Possibile definire frazione doc in pipeline Hi-Res (fraction_hires), e la parte restante
  è pipeline Fast. In base a ciò, un tot di doc usa c_page=0.01, e un tot usa c_page=0.001.

NOTA: questo script produce in output una serie di "print" di debug
      per evidenziare come viene composto il costo finale.
"""

# Importiamo le funzioni dal "primo script" (chat usage)
# Nomi di moduli/funzioni puramente dimostrativi:
from app.send_message import (
    calculate_history_tokens,
    build_history_text,
    cost_of_message,
)

# Importiamo le funzioni dal "secondo script" (ingestion, file kbox):
from app.upload_file_in_kbox import (
    tokens_for_resolution,
    cost_upload_pdf,
    cost_upload_image,
    cost_upload_video
)


def cost_monthly_user_debug(
    # Parametri generali per la chat
    n_chat: int,            # Numero medio di chat create al mese
    n_msg_per_chat: int,    # Numero di messaggi per chat
    user_tokens_per_msg: float,  # Token input medi dell'utente
    n_kbox_per_msg: float,       # KBox consultate in media
    r_per_kbox: float,           # Risultati per KBox
    chunk_size_retrieval: float, # Token chunk di retrieval
    out_tokens_per_msg: float,   # Token output medi (risposta)
    fraction_4o: float,          # Frazione di messaggi GPT-4o vs. (1 - fraction_4o) su GPT-4o Mini

    # Costi GPT-4o per chat
    p_in_4o: float,             # costo input GPT-4o (es. 0.005)
    p_out_4o: float,            # costo output GPT-4o (es. 0.015)
    # Costi GPT-4o Mini per chat
    p_in_mini: float,           # costo input GPT-4o Mini (es. 0.00015)
    p_out_mini: float,          # costo output GPT-4o Mini (es. 0.00060)

    # Costi retrieval e store per un messaggio
    c_retrieval: float,         # costo retrieval fisso (embedding query + ricerche)
    c_store: float,             # costo store fisso per messaggio

    # Parametri ingestion
    n_doc: int,                 # documenti caricati al mese
    n_img: int,                 # immagini caricate al mese
    n_vid: int,                 # video caricati al mese

    # FRAZIONE di doc che usa pipeline hi-res vs fast
    fraction_hires: float,      # es. 0.6 => 60% doc usano hi-res, 40% doc usano fast
    # costi "hi-res" e "fast"
    c_page_hires: float,       # 0.01
    c_page_fast: float,        # 0.001

    t_total_doc: float,        # token totali medi per doc
    barT_chunk_doc: float,
    c_embed_doc: float,
    c_db_chunk_doc: float,
    pages_per_doc: int,

    # FRAZIONE di immagini/video che usa GPT-4o vs GPT-4o Mini:
    fraction_4o_imgvid: float,  # es. 0.3 => 30% di img/video caption con GPT-4o, 70% con mini
    # costi GPT-4o (immagini/video):
    p_in_4o_imgvid: float,      # es. 0.005
    p_out_4o_imgvid: float,     # es. 0.015
    # costi GPT-4o Mini (immagini/video):
    p_in_mini_imgvid: float,    # es. 0.00015
    p_out_mini_imgvid: float,   # es. 0.00060

    # param immagine
    img_width: int,
    img_height: int,
    t_descr_img: float,
    c_embed_img: float,
    c_db_chunk_img: float,

    # param video
    dur_sec_vid: float,
    sampling_sec_vid: float,
    vid_width: int,
    vid_height: int,
    t_descr_vid_frame: float,
    c_embed_vid: float,
    c_db_chunk_vid: float,
    t_descr_vid_total: float,  # token totali di tutte le caption fuse

    # Parametri storage
    gb_stored_user: float,   # GB memorizzati dall'utente
    c_gb_month: float        # costo per GB/mese (es. 0.25)
) -> float:
    """
    Calcola il costo mensile di un utente, con debug di tutte le voci (chat, ingestion, storage).

    - Chat: frazione fraction_4o di messaggi su GPT-4o, il resto su GPT-4o Mini
    - Documenti: frazione fraction_hires (0.01 $/page), e (1 - fraction_hires) su 0.001 $/page
    - Immagini/Video: frazione fraction_4o_imgvid su GPT-4o, resto su GPT-4o Mini

    Ritorna un float in dollari.
    """
    print("=== DEBUG COST MONTHLY USER (MIX GPT-4o/Mini su Chat e su Immagini/Video) ===")
    print("PARAMETRI CHAT / FRAZIONE GPT-4o:")
    print(f"  n_chat = {n_chat}, n_msg_per_chat = {n_msg_per_chat}, fraction_4o (chat) = {fraction_4o}")
    print(f"  => GPT-4o: p_in_4o={p_in_4o}, p_out_4o={p_out_4o}")
    print(f"  => GPT-4o Mini: p_in_mini={p_in_mini}, p_out_mini={p_out_mini}")
    print()

    print("PARAMETRI INGESTION (DOC):")
    print(f"  n_doc={n_doc}, fraction_hires={fraction_hires}, c_page_hires={c_page_hires}, c_page_fast={c_page_fast}")
    print(f"  t_total_doc={t_total_doc}, barT_chunk_doc={barT_chunk_doc}, pages_per_doc={pages_per_doc}")
    print()

    print("PARAMETRI INGESTION (IMG/VID):")
    print(f"  n_img={n_img}, n_vid={n_vid}, fraction_4o_imgvid={fraction_4o_imgvid}")
    print(f"  => GPT-4o (img/vid): p_in_4o_imgvid={p_in_4o_imgvid}, p_out_4o_imgvid={p_out_4o_imgvid}")
    print(f"  => GPT-4o Mini (img/vid): p_in_mini_imgvid={p_in_mini_imgvid}, p_out_mini_imgvid={p_out_mini_imgvid}")
    print(f"  => Immagine: width={img_width}, height={img_height}, t_descr_img={t_descr_img}, c_embed_img={c_embed_img}, c_db_chunk_img={c_db_chunk_img}")
    print(f"  => Video: dur_sec_vid={dur_sec_vid}, sampling_sec_vid={sampling_sec_vid}, width={vid_width}, height={vid_height}, t_descr_vid_frame={t_descr_vid_frame}")
    print(f"            c_embed_vid={c_embed_vid}, c_db_chunk_vid={c_db_chunk_vid}, t_descr_vid_total={t_descr_vid_total}")
    print()

    print("PARAMETRI STORAGE:")
    print(f"  gb_stored_user={gb_stored_user} GB, c_gb_month={c_gb_month}")
    print("=========================================================\n")

    #
    # 1) Calcolo COSTO CHAT: mescoliamo fraction_4o con GPT-4o e (1 - fraction_4o) con GPT-4o Mini
    #
    from math import ceil

    total_msgs = n_chat * n_msg_per_chat

    print("=== DEBUG: COSTO CHAT ===")
    # 1a) calcoliamo costo di un messaggio GPT-4o puro
    cost_one_message_4o = cost_of_message(
        max_pairs=25,
        avg_tokens_per_message=100,  # fissi ipotetici
        user_tokens=user_tokens_per_msg,
        n_kbox=n_kbox_per_msg,
        r_per_kbox=r_per_kbox,
        chunk_size=chunk_size_retrieval,
        out_tokens=out_tokens_per_msg,
        p_in=p_in_4o,
        p_out=p_out_4o,
        c_retrieval=c_retrieval,
        c_store=c_store
    )
    # 1b) calcoliamo costo di un messaggio GPT-4o Mini puro
    cost_one_message_mini = cost_of_message(
        max_pairs=25,
        avg_tokens_per_message=100,  # ipotetici
        user_tokens=user_tokens_per_msg,
        n_kbox=n_kbox_per_msg,
        r_per_kbox=r_per_kbox,
        chunk_size=chunk_size_retrieval,
        out_tokens=out_tokens_per_msg,
        p_in=p_in_mini,
        p_out=p_out_mini,
        c_retrieval=c_retrieval,
        c_store=c_store
    )
    fraction_mini = 1.0 - fraction_4o
    cost_one_message_mixed = fraction_4o*cost_one_message_4o + fraction_mini*cost_one_message_mini
    cost_chat_month = total_msgs * cost_one_message_mixed

    print(f"   cost_one_message_4o   = {cost_one_message_4o:.6f}  (GPT-4o puro)")
    print(f"   cost_one_message_mini = {cost_one_message_mini:.6f}  (GPT-4o Mini puro)")
    print(f"   fraction_4o = {fraction_4o}, fraction_mini = {fraction_mini}")
    print(f"   => cost_one_message_mixed = {cost_one_message_mixed:.6f}")
    print(f"   total_msgs = {total_msgs}, => cost_chat_month = {cost_chat_month:.6f}\n")

    #
    # 2) Costo ingestion
    #
    cost_ingestion = 0.0
    print("=== DEBUG: COSTO INGESTION (DOC, IMG, VID) ===")

    # 2a) Documenti
    print(f" -> Caricamento di {n_doc} doc totali.")
    n_doc_hires = int(round(n_doc * fraction_hires))
    n_doc_fast  = n_doc - n_doc_hires
    print(f"    fraction_hires = {fraction_hires} => n_doc_hires={n_doc_hires}, n_doc_fast={n_doc_fast}")

    # Carichiamo doc hires
    for i in range(n_doc_hires):
        cost_doc = cost_upload_pdf(
            n_pages=pages_per_doc,
            c_page=c_page_hires,   # 0.01
            t_total=t_total_doc,
            barT_chunk=barT_chunk_doc,
            c_embed=c_embed_doc,
            c_db_chunk=c_db_chunk_doc
        )
        cost_ingestion += cost_doc
        print(f"    doc hires #{i+1}: cost_doc = {cost_doc:.6f} $")

    # Carichiamo doc fast
    for j in range(n_doc_fast):
        cost_doc = cost_upload_pdf(
            n_pages=pages_per_doc,
            c_page=c_page_fast,  # 0.001
            t_total=t_total_doc,
            barT_chunk=barT_chunk_doc,
            c_embed=c_embed_doc,
            c_db_chunk=c_db_chunk_doc
        )
        cost_ingestion += cost_doc
        print(f"    doc fast #{j+1}: cost_doc = {cost_doc:.6f} $")

    # 2b) Immagini
    print(f"\n -> Caricamento di {n_img} immagini.")
    n_img_gpt4o = int(round(n_img * fraction_4o_imgvid))
    n_img_mini  = n_img - n_img_gpt4o
    print(f"    fraction_4o_imgvid={fraction_4o_imgvid} => n_img_gpt4o={n_img_gpt4o}, n_img_mini={n_img_mini}")

    for i in range(n_img_gpt4o):
        cost_img = cost_upload_image(
            width=img_width,
            height=img_height,
            t_descr=t_descr_img,
            p_in=p_in_4o_imgvid,
            p_out=p_out_4o_imgvid,
            c_embed=c_embed_img,
            c_db_chunk=c_db_chunk_img
        )
        cost_ingestion += cost_img
        print(f"    img GPT-4o #{i+1}: cost_img = {cost_img:.6f} $")

    for j in range(n_img_mini):
        cost_img = cost_upload_image(
            width=img_width,
            height=img_height,
            t_descr=t_descr_img,
            p_in=p_in_mini_imgvid,
            p_out=p_out_mini_imgvid,
            c_embed=c_embed_img,
            c_db_chunk=c_db_chunk_img
        )
        cost_ingestion += cost_img
        print(f"    img GPT-4o Mini #{j+1}: cost_img = {cost_img:.6f} $")

    # 2c) Video
    print(f"\n -> Caricamento di {n_vid} video.")
    n_vid_gpt4o = int(round(n_vid * fraction_4o_imgvid))
    n_vid_mini  = n_vid - n_vid_gpt4o
    print(f"    fraction_4o_imgvid={fraction_4o_imgvid} => n_vid_gpt4o={n_vid_gpt4o}, n_vid_mini={n_vid_mini}")

    for i in range(n_vid_gpt4o):
        cost_vid = cost_upload_video(
            duration_sec=dur_sec_vid,
            sampling_sec=sampling_sec_vid,
            width=vid_width,
            height=vid_height,
            t_descr=t_descr_vid_frame,
            p_in=p_in_4o_imgvid,
            p_out=p_out_4o_imgvid,
            c_embed=c_embed_vid,
            c_db_chunk=c_db_chunk_vid,
            t_descr_total=t_descr_vid_total
        )
        cost_ingestion += cost_vid
        print(f"    video GPT-4o #{i+1}: cost_vid = {cost_vid:.6f} $")

    for j in range(n_vid_mini):
        cost_vid = cost_upload_video(
            duration_sec=dur_sec_vid,
            sampling_sec=sampling_sec_vid,
            width=vid_width,
            height=vid_height,
            t_descr=t_descr_vid_frame,
            p_in=p_in_mini_imgvid,
            p_out=p_out_mini_imgvid,
            c_embed=c_embed_vid,
            c_db_chunk=c_db_chunk_vid,
            t_descr_total=t_descr_vid_total
        )
        cost_ingestion += cost_vid
        print(f"    video GPT-4o Mini #{j+1}: cost_vid = {cost_vid:.6f} $")

    print(f"\n => cost_ingestion (totale) = {cost_ingestion:.6f} $\n")

    #
    # 3) Costo storage mensile
    #
    cost_storage = gb_stored_user * c_gb_month
    print("=== DEBUG: COST STORAGE ===")
    print(f"  gb_stored_user={gb_stored_user:.2f} GB, c_gb_month={c_gb_month:.3f} $/GB/mese")
    print(f"  => cost_storage = {cost_storage:.6f} $\n")

    # Somma finale
    cost_month_user = cost_chat_month + cost_ingestion + cost_storage
    print("=== RISULTATO FINALE ===")
    print(f"  Costo Chat:      {cost_chat_month:.6f} $")
    print(f"  Costo Ingestion: {cost_ingestion:.6f} $")
    print(f"  Costo Storage:   {cost_storage:.6f} $")
    print("---------------------------------")
    print(f" ==> COSTO TOTALE MENSILE UTENTE = {cost_month_user:.6f} $\n")

    return cost_month_user


if __name__ == "__main__":
    # Esempio di parametri tipici + debug
    monthly_cost_debug = cost_monthly_user_debug(
        # chat usage
        n_chat=8,
        n_msg_per_chat=20,
        user_tokens_per_msg=100,
        n_kbox_per_msg=1.5,
        r_per_kbox=5,
        chunk_size_retrieval=300,
        out_tokens_per_msg=300,
        fraction_4o=0.3,  # 20% GPT-4o, 80% GPT-4o Mini
        p_in_4o=0.005,
        p_out_4o=0.015,
        p_in_mini=0.00015,
        p_out_mini=0.00060,
        c_retrieval=1e-5,
        c_store=1e-5,

        # ingestion
        n_doc=8,         # tot doc
        n_img=2,         # tot img
        n_vid=1,         # tot video

        fraction_hires=0.5,      # 50% doc hi-res, 50% doc fast
        c_page_hires=0.01,
        c_page_fast=0.001,
        t_total_doc=5000,
        barT_chunk_doc=500,
        c_embed_doc=0.00002,
        c_db_chunk_doc=7.5e-6,
        pages_per_doc=10,

        fraction_4o_imgvid=0.3,  # 30% GPT-4o (img/video), 70% GPT-4o Mini
        p_in_4o_imgvid=0.005,
        p_out_4o_imgvid=0.015,
        p_in_mini_imgvid=0.00015,
        p_out_mini_imgvid=0.00060,

        img_width=512,
        img_height=512,
        t_descr_img=100,
        c_embed_img=0.00002,
        c_db_chunk_img=7.5e-6,

        dur_sec_vid=120,
        sampling_sec_vid=10,
        vid_width=512,
        vid_height=512,
        t_descr_vid_frame=50,
        c_embed_vid=0.00002,
        c_db_chunk_vid=7.5e-6,
        t_descr_vid_total=600,

        gb_stored_user=0.1,  # 0.1 GB
        c_gb_month=0.25
    )
    print(f"Costo mensile utente (DEBUG, doc hi-res/fast, mix GPT-4o & Mini anche su img/video): {monthly_cost_debug:.6f} $")
