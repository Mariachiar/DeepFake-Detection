"""
Questo script implementa un sistema di rilevamento facciale in tempo reale basato sul modello YuNet,
ed è stato adattato per supportare applicazioni di visione artificiale su flussi video dinamici,
come quelli provenienti da webcam o videochiamate.

YuNet è un modello di deep learning anchor-free per face detection, ottimizzato per lavorare
in tempo reale anche su dispositivi a basse risorse computazionali (es. CPU, edge device).
Il modello viene eseguito tramite l'interfaccia `cv2.FaceDetectorYN` introdotta in Opencv2 4.10,
che permette l'inferenza diretta da modelli in formato ONNX.

Nel contesto di questa tesi, lo script rappresenta la **prima fase fondamentale di preprocessing**
per una pipeline di rilevamento di contenuti manipolati (deepfake) da videostream.
Nello specifico, si occupa di:
- acquisire frame da webcam o da immagine statica;
- rilevare i volti presenti nel frame;
- visualizzare e opzionalmente salvare le bounding box e i landmark facciali rilevati;
- fornire in output i dati strutturati (coordinate, confidenza) necessari per la fase successiva di tracciamento e analisi.

L’utilizzo di un modello efficiente come YuNet consente di operare a frame rate elevati (oltre 10–15 FPS)
anche su macchine non dotate di GPU, rendendo questa soluzione ideale per applicazioni distribuite
o in tempo reale.

Questo script può essere esteso per:
- integrare un sistema di tracciamento (es. ByteTrack) con ID persistenti;
- aggregare clip temporali per l’analisi di coerenza dinamica;
- passare i volti ritagliati a un classificatore per la rilevazione di manipolazioni deepfake.

Nel complesso, rappresenta un **blocco funzionale autonomo e riutilizzabile**, orientato
all’elaborazione efficiente di stream video facciali in contesti applicativi avanzati.
"""

import cv2                   # OpenCV per elaborazione video e grafica
import numpy as np           # Array e operazioni numeriche
import time                  # Per misurare tempo e FPS
from yunet.yunet import YuNet                    # Modello di face detection
from ByteTrack.byte_tracker import BYTETracker, STrack  # Tracciamento multi-oggetto
import argparse              # Gestione argomenti da terminale
from argparse import Namespace
import os                   # Gestione file e cartelle
import pandas as pd
import matplotlib.pyplot as plt

# === Configurazioni iniziali ===
input_size = [320, 320]  # Risoluzione di input per YuNet
conf_threshold = 0.9     # Soglia minima di confidenza per rilevamento volto
nms_threshold = 0.3      # Soglia per suppression di bounding box sovrapposte
top_k = 5000             # Numero massimo di box da considerare prima di NMS

# === Definizione dei back-end per inferenza (CPU, GPU ecc.) ===
backend_target_pairs = [
    [cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU],
    [cv2.dnn.DNN_BACKEND_CUDA,   cv2.dnn.DNN_TARGET_CUDA],
    [cv2.dnn.DNN_BACKEND_CUDA,   cv2.dnn.DNN_TARGET_CUDA_FP16],
    [cv2.dnn.DNN_BACKEND_TIMVX,  cv2.dnn.DNN_TARGET_NPU],
    [cv2.dnn.DNN_BACKEND_CANN,   cv2.dnn.DNN_TARGET_NPU]
]


parser = argparse.ArgumentParser(description='Demo preprocessing con YuNet + ByteTrack.')
parser.add_argument('--input', '-i', type=str,
                    help='Usage: Set input to a certain image, omit if using camera.')
parser.add_argument('--model', '-m', type=str, default='C:\\Users\\maria\\Desktop\\deepfake\\preprocessing\\yunet\\face_detection_yunet_2023mar.onnx',
                    help="Usage: Set model type, defaults to 'face_detection_yunet_2023mar.onnx'.")
parser.add_argument('--backend_target', '-bt', type=int, default=0,
                    help='''Choose one of the backend-target pair to run this demo:
                        {:d}: (default) Opencv2 implementation + CPU,
                        {:d}: CUDA + GPU (CUDA),
                        {:d}: CUDA + GPU (CUDA FP16),
                        {:d}: TIM-VX + NPU,
                        {:d}: CANN + NPU
                    '''.format(*[x for x in range(len(backend_target_pairs))]))
parser.add_argument('--conf_threshold', type=float, default=0.9,
                    help='Usage: Set the minimum needed confidence for the model to identify a face, defauts to 0.9. Smaller values may result in faster detection, but will limit accuracy. Filter out faces of confidence < conf_threshold.')
parser.add_argument('--nms_threshold', type=float, default=0.3,
                    help='Usage: Suppress bounding boxes of iou >= nms_threshold. Default = 0.3.')
parser.add_argument('--top_k', type=int, default=5000,
                    help='Usage: Keep top_k bounding boxes before NMS.')
parser.add_argument('--save', '-s', action='store_true',
                    help='Usage: Specify to save file with results (i.e. bounding box, confidence level). Invalid in case of camera input.')
parser.add_argument('--vis', '-v', action='store_true',
                    help='Usage: Specify to open a new window to show results. Invalid in case of camera input.')
args = parser.parse_args()

if __name__ == '__main__':
    backend_id = backend_target_pairs[args.backend_target][0] #Estrae il backend/target scelto dall’utente dalla lista backend_target_pairs.
    target_id = backend_target_pairs[args.backend_target][1]

# === Inizializza YuNet ===
face_detector = YuNet(
    modelPath=args.model,
    inputSize=input_size,
    confThreshold=conf_threshold,
    nmsThreshold=nms_threshold,
    topK=top_k,
    backendId=backend_id,
    targetId=target_id
)

# === Inizializza ByteTrack ===
tracker_args = {
    "track_thresh": 0.5, #Soglia di confidenza minima affinché una nuova detection venga considerata per il tracciamento.
    "match_thresh": 0.8, #Soglia di similarità per associare una detection a una traccia esistente. È usata nella funzione di assegnamento (LAPJV/Hungarian) basata sulla matrice dei costi (di solito IoU o distanza di Mahalanobis).
    "track_buffer": 10000, #Numero massimo di frame consecutivi in cui una traccia può rimanere “persa” (non associata a una nuova detection) prima di essere eliminata. 30 frame a 15 FPS ≈ 2 secondi di tolleranza.
    "frame_rate": 15, #Frame rate del video di input, usato per adattare i parametri temporali (es. durata della buffer, predizione del filtro di Kalman, ecc.).
    "mot20": False  # flag usato per decidere se usare una soglia diversa per l'algoritmo di assegnamento nel tracking (è specifico per il dataset MOT20, con più affollamento). Nel tuo caso non serve, ma è incluso nel codice generico di ByteTrack.
}
tracker_args = Namespace(**tracker_args)
tracker = BYTETracker(args=tracker_args, frame_rate=15)

# === Webcam ===
cap = cv2.VideoCapture(0)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
face_detector.setInputSize([w, h])

frame_id = 0
clip_buffer = {}  # {track_id: [face1, face2, ..., face8]}
clip_start_times = {}  # track_id → timestamp di quando ha accumulato il primo frame

# === Init buffer per ogni ID ===
CLIP_LENGTH = 8
CLIP_STEP = 4
CLIP_DIR = "clips"
clip_logs = []  # Log per CSV
os.makedirs(CLIP_DIR, exist_ok=True)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    start = time.time()

    # === Rilevamento volti ===
    detections = face_detector.infer(frame)
    faces = []

    # Per ogni rilevamento del volto effettuato da YuNet:
    for det in detections:
        x, y, w_box, h_box = det[:4]  # Estrae le coordinate della bounding box (top-left x/y, larghezza, altezza).
        conf = det[-1]                # Estrae il punteggio di confidenza della detection (quanto è sicuro il modello).
        tlwh = [x, y, w_box, h_box]   # Converte i dati in formato TLWH (Top-Left X/Y, Width, Height), richiesto dal tracker.
        
        # Inizializza un oggetto STrack (Single Track) per ciascun volto rilevato,
        # contenente bounding box e confidenza. Questo oggetto rappresenta un'ipotesi iniziale di tracciamento,
        # che verrà poi gestita dal tracker (BYTETracker) per mantenere la coerenza temporale.
        faces.append(STrack(tlwh, score=conf))


    # === Tracciamento ===
    img_h, img_w = frame.shape[:2] ## (altezza, larghezza) dimensione del frame corrente, così come è stato catturato dalla webcam
    img_info = (img_h, img_w)
    img_size = (img_w, img_h)  # nel formato richiesto dal tracker

    online_targets = tracker.update(faces, img_info, img_size)


    # === Visualizzazione ===
    for track in online_targets:
        if not track.is_activated:
            continue

        tlwh = track.tlwh
        track_id = track.track_id
        x1, y1, w_box, h_box = map(int, tlwh)
        cv2.rectangle(frame, (x1, y1), (x1 + w_box, y1 + h_box), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 1)

        # === Ritaglia il volto dal frame ===
        face = frame[y1:y1 + h_box, x1:x1 + w_box]
        if face.size == 0:
            continue  # Salta se bounding box fuori dai limiti o vuota

        # === Resize e normalizzazione ===
        face_resized = cv2.resize(face, (224, 224))
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)

        # === Aggiungi il volto normalizzato al buffer di quel track_id ===
        if track_id not in clip_buffer:
            clip_buffer[track_id] = []
            clip_start_times[track_id] = time.time()  # salva il tempo di inizio accumulo

        clip_buffer[track_id].append(face_rgb)

        # === Se abbiamo almeno 8 frame, salva la clip ===
        if len(clip_buffer[track_id]) >= CLIP_LENGTH:
            elapsed_time = time.time() - clip_start_times[track_id]  # tempo totale per raccogliere la clip
            fps_clip = CLIP_LENGTH / elapsed_time if elapsed_time > 0 else 0  # calcolo FPS effettivi per la clip

            clip = np.stack(clip_buffer[track_id][:CLIP_LENGTH])  # (8, 224, 224, 3)
            save_path = os.path.join(CLIP_DIR, f"track{track_id}_frame{frame_id}.npy")
            np.save(save_path, clip)

            print(f"✔️ Clip salvata: {save_path}")
            print(f"⏱️ Tempo raccolta: {elapsed_time:.2f} s | FPS medi: {fps_clip:.2f} | track_id = {track_id}")

            # Salva log in memoria
            clip_logs.append({
                "frame_id": frame_id,
                "track_id": track_id,
                "elapsed_time": elapsed_time,
                "fps_clip": fps_clip
            })

            # === Mantieni gli ultimi 4 frame per slide window ===
            clip_buffer[track_id] = clip_buffer[track_id][CLIP_STEP:]
            clip_start_times[track_id] = time.time()  # resetta il tempo per la prossima clip


            # === Mantieni gli ultimi 4 frame per slide window ===
            clip_buffer[track_id] = clip_buffer[track_id][CLIP_STEP:]

    end = time.time()
    fps = 1.0 / (end - start)
    cv2.putText(
        frame,                      # immagine su cui scrivere
        f"FPS: {fps:.2f}",          # testo da scrivere, con 2 cifre decimali
        (10, 20),                   # coordinate (x, y) del punto di partenza
        cv2.FONT_HERSHEY_SIMPLEX,  # font
        0.5,                        # scala del testo
        (0, 0, 255),                # colore in BGR → rosso
        1                          # spessore della linea
    )


    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()             # Ferma la cattura della webcam
cv2.destroyAllWindows()   # Chiude tutte le finestre aperte da OpenCV

# === Salva CSV e genera grafico ===
log_df = pd.DataFrame(clip_logs)
log_path = "fps_log.csv"
log_df.to_csv(log_path, index=False)

plt.figure(figsize=(10, 5))
plt.plot(log_df["frame_id"], log_df["fps_clip"], marker='o', linestyle='-', color='blue')
plt.title("Andamento FPS medi per clip salvata")
plt.xlabel("Frame ID")
plt.ylabel("FPS medi (clip)")
plt.grid(True)
plt.tight_layout()
plt.savefig("fps_plot.png")
plt.show()


