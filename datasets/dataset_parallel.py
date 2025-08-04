import os
# Mantiene il workaround per l'errore OMP
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
import time
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import torch
from argparse import Namespace
import sys
from pathlib import Path

# Importa i moduli custom (assicurati che il percorso sia corretto)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from preprocessing.yunet.yunet import YuNet
from preprocessing.ByteTrack.byte_tracker import BYTETracker, STrack, TrackState

import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

# Modulo per il parallelismo basato su thread
from concurrent.futures import ThreadPoolExecutor, as_completed # as_completed è utile per raccogliere i risultati
from preprocessing.libreface.libreface_adapter import  get_au_from_face_ndarray, _initialize_au_model as libreface_init_au_model
# === Sezione: Importazione e Inizializzazione per l'estrazione delle Action Units (AU) ===
try:
    from preprocessing.libreface.libreface_adapter import get_au_from_face_ndarray, _initialize_au_model as libreface_init_au_model
    print("LibreFace importato correttamente per l'estrazione delle AU.")
    _libreface_available = True
except ImportError:
    print("ATTENZIONE: libreface/libreface_adapter.py o get_au_from_face_ndarray non trovato.")
    print("Verrà utilizzata una funzione placeholder per l'estrazione delle AU (simulazione batch).")
    def get_au_from_face_ndarray(face_rgbs_batch):
        # Placeholder che restituisce un dizionario vuoto per ogni volto nel batch
        # O puoi simulare dei valori casuali come nell'esempio precedente se preferisci
        batch_results = []
        for _ in face_rgbs_batch:
            batch_results.append({
                "AU01": np.random.rand(), "AU02": np.random.rand(), "AU04": np.random.rand(),
                "AU06": np.random.rand(), "AU07": np.random.rand(), "AU10": np.random.rand(),
                "AU12": np.random.rand(), "AU14": np.random.rand(), "AU15": np.random.rand(),
                "AU17": np.random.rand(), "AU23": np.random.rand(), "AU24": np.random.rand()
            })
        return batch_results
    _libreface_available = False # Flag per sapere se LibreFace è disponibile


# === Sezione: Configurazioni Globali della Pipeline ===
CLIP_LENGTH = 8
CLIP_STEP = 4
CLIP_SIZE = (224, 224) # Dimensione a cui ridimensionare il volto per MediaPipe e le clip
AU_CLIP_LENGTH = 8 # Se vuoi gestire anche le AU in clip
AU_CLIP_STEP = 4

CLIP_DIR = "clips"
os.makedirs(CLIP_DIR, exist_ok=True)

input_size = [320, 320] # Default per YuNet
backend_target_pairs = [
    [cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU],
    [cv2.dnn.DNN_BACKEND_CUDA, cv2.dnn.DNN_TARGET_CUDA]
]

# === Sezione: Parsing degli Argomenti da Riga di Comando ===
parser = argparse.ArgumentParser(description="Real-time Multi-person Deepfake Preprocessing Pipeline")
parser.add_argument('--model', '-m', type=str,
                    default='C:\\Users\\maria\\Desktop\\deepfake\\preprocessing\\yunet\\face_detection_yunet_2023mar.onnx',
                    help='Percorso al modello ONNX di YuNet (per il rilevamento facciale).')
parser.add_argument('--input', type=str, default=None, help='Percorso a un video specifico o lascia vuoto per webcam')
parser.add_argument('--backend_target', '-bt', type=int, default=0,
                    help='Backend e target per YuNet. 0: OpenCV-CPU (default), 1: CUDA-GPU.')
parser.add_argument('--mode', type=str, choices=['save', 'memory'], default='save',
                    help='Modalità di gestione delle clip: "save" per salvare su disco, "memory" per mantenere in RAM.')
parser.add_argument('--vis', '-v', action='store_true',
                    help='Abilita la visualizzazione in tempo reale con bounding box, ID, landmark e FPS.')
parser.add_argument('--num_workers_per_frame', type=int, default=os.cpu_count() or 1,
                    help='Numero di thread da usare per elaborare i volti in parallelo (per MediaPipe).')
parser.add_argument('--show_faces', action='store_true',
                    help='Mostra finestre separate per ogni volto ritagliato con landmarks.')
args = parser.parse_args()


# === Blocco di esecuzione principale ===
if __name__ == "__main__":
    # --- Schermata di caricamento iniziale per feedback utente ---
    cv2.namedWindow("Tracking & Facial Analysis", cv2.WINDOW_NORMAL)
    loading_img = np.zeros((480, 640, 3), dtype=np.uint8)

    cv2.putText(loading_img, "Loading models... Please wait.", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imshow("Tracking & Facial Analysis", loading_img)
    cv2.waitKey(1) # Mostra la finestra per un breve momento

    # --- Inizializzazione dei modelli e componenti pesanti ---

    # 1. Inizializzazione LibreFace AU (se disponibile)
    if _libreface_available:
        print("Forzatura dell'inizializzazione di LibreFace AU...")
        try:
            libreface_init_au_model() # Chiamata esplicita alla funzione di inizializzazione
            cv2.putText(loading_img, "LibreFace AU loaded.", (50, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.imshow("Tracking & Facial Analysis", loading_img)
            cv2.waitKey(1)
        except Exception as e:
            print(f"ERRORE: Inizializzazione LibreFace AU fallita: {e}")
            cv2.putText(loading_img, "LibreFace AU FAILED!", (50, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            cv2.imshow("Tracking & Facial Analysis", loading_img)
            cv2.waitKey(0) # Aspetta una pressione di tasto prima di chiudere
            sys.exit(1)
    else:
        cv2.putText(loading_img, "LibreFace AU (placeholder).", (50, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 150, 255), 1)
        cv2.imshow("Tracking & Facial Analysis", loading_img)
        cv2.waitKey(1)

    # 2. Inizializzazione YuNet
    backend_id, target_id = backend_target_pairs[args.backend_target]
    print(f"DEBUG: Numero massimo di thread configurati per l'executor: {args.num_workers_per_frame}")
    face_detector = YuNet(
        modelPath=args.model,
        inputSize=input_size,
        confThreshold=0.9,
        nmsThreshold=0.3,
        topK=5000,
        backendId=backend_id,
        targetId=target_id
    )
    cv2.putText(loading_img, "YuNet loaded.", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    cv2.imshow("Tracking & Facial Analysis", loading_img)
    cv2.waitKey(1)

    # 3. Inizializzazione ByteTrack
    tracker_args = Namespace(track_thresh=0.5, match_thresh=0.8, track_buffer=30, frame_rate=15, mot20=False)
    tracker = BYTETracker(args=tracker_args, frame_rate=15)
    cv2.putText(loading_img, "ByteTrack initialized.", (50, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    cv2.imshow("Tracking & Facial Analysis", loading_img)
    cv2.waitKey(1)

    # NOTA BENE: MediaPipe Face Mesh NON viene inizializzato qui globalmente per il multi-threading
    # Verrà inizializzato all'interno di _process_face_mesh_for_thread
    cv2.putText(loading_img, "MediaPipe FaceMesh will be loaded per-thread.", (50, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    cv2.imshow("Tracking & Facial Analysis", loading_img)
    cv2.waitKey(1)

    # 5. Inizializzazione del ThreadPoolExecutor
    executor = ThreadPoolExecutor(max_workers=args.num_workers_per_frame)
    # Lista per tenere traccia dei futuri sottomessi
    # future_tasks conterrà tuple: (future_obj, track_id, original_face_rgb, original_bbox, original_bbox_dims, aus_data)
    future_tasks = [] 
    cv2.putText(loading_img, "ThreadPoolExecutor ready.", (50, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    cv2.imshow("Tracking & Facial Analysis", loading_img)
    cv2.waitKey(1)

       # 4. Inizializzazione MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # === Cattura Video dalla Webcam (Spostata qui dopo tutte le inizializzazioni) ===
    input_source = args.input
    if input_source is None:
        cap = cv2.VideoCapture(0)
    else:
        input_path = Path(input_source)
        if not input_path.exists():
            print(f"Errore: il file {input_source} non esiste.")
            sys.exit(1)
        cap = cv2.VideoCapture(str(input_path))

    if not cap.isOpened():
        print("Errore: impossibile aprire il video o la webcam.")
        sys.exit(1)

    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    face_detector.setInputSize([w, h])

    cv2.putText(loading_img, "Video/Webcam initialized.", (50, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    cv2.imshow("Tracking & Facial Analysis", loading_img)
    cv2.waitKey(1000) # Mantieni la schermata di caricamento un po' di più alla fine

    # Chiudi la finestra di caricamento prima di avviare il flusso principale
    cv2.destroyWindow("Tracking & Facial Analysis")

    # === Variabili per la Gestione delle Clip e il Logging ===
    frame_id = 0
    clip_buffer = {}
    au_buffer = {} # Buffer per le clip AU
    clip_start_times = {} # Tempo di inizio per le clip immagine
    au_clip_start_times = {} # Tempo di inizio per le clip AU
    clip_logs = []
    clips_in_ram = []

    print(f"Acquisizione video avviata. Premi 'ESC' per uscire.")

    # === Funzione helper per l'elaborazione di MediaPipe (ORA CREA UNA NUOVA ISTANZA) ===
    # Questo è il cuore della correzione per il threading
    def _process_face_mesh_for_thread(face_rgb_input):
        # Ogni thread avrà la sua istanza di FaceMesh
        with mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, # Usiamo True perché processiamo singole immagini
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5 # Questo non ha effetto con static_image_mode=True
        ) as face_mesh_detector_local:
            results_mesh_thread = face_mesh_detector_local.process(face_rgb_input)
            return results_mesh_thread.multi_face_landmarks[0] if results_mesh_thread.multi_face_landmarks else None

    # === Ciclo Principale di Elaborazione del Video ===
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Fine del flusso video o errore di lettura del frame.")
                break

            frame_id += 1
            start_frame_time = time.time()

            # 1. Rilevamento facciale con YuNet
            detections = face_detector.infer(frame)
            faces_detected_for_tracking = [STrack(det[:4], score=det[-1]) for det in detections]

            img_h, img_w = frame.shape[:2]

            # 2. Tracciamento multi-volto con ByteTrack
            online_targets = tracker.update(faces_detected_for_tracking, (img_h, img_w), (img_w, img_h))

            current_frame_faces_data_for_batch = [] # Dati per l'AU batch e per sottomettere i futuri
            faces_for_mediapipe_vis = [] # Dati da usare per la visualizzazione dopo che i futuri sono completati

            # Raccogli i dati di tutti i volti per il frame corrente
            for track in online_targets:
                if not track.is_activated or track.state == TrackState.Lost:
                    continue

                track_id = track.track_id
                x1, y1, w_box, h_box = map(int, track.tlwh)
                
                # --- INIZIO MODIFICA: Aggiunta padding al bounding box ---
                padding_factor = 0.15 # Esempio: 15% di padding su larghezza/altezza
                pad_x = int(w_box * padding_factor)
                pad_y = int(h_box * padding_factor)

                x1_padded = max(0, x1 - pad_x)
                y1_padded = max(0, y1 - pad_y)
                x2_padded = min(x1 + w_box + pad_x, frame.shape[1])
                y2_padded = min(y1 + h_box + pad_y, frame.shape[0])

                w_padded = x2_padded - x1_padded
                h_padded = y2_padded - y1_padded
                # --- FINE MODIFICA ---

                # Utilizza il bounding box con padding per il crop
                face_cropped = frame[y1_padded:y2_padded, x1_padded:x2_padded]
                
                if face_cropped.size == 0 or w_padded <= 0 or h_padded <= 0:
                    continue
                
                face_resized = cv2.resize(face_cropped, CLIP_SIZE) # CLIP_SIZE è ancora (224, 224)
                face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)

                current_frame_faces_data_for_batch.append({
                    "track_id": track_id,
                    "face_rgb": face_rgb,
                    # Passa le coordinate del bounding box padded e le dimensioni padded
                    "bbox": (x1_padded, y1_padded, x2_padded, y2_padded), 
                    "original_bbox_dims": (w_padded, h_padded) # Queste sono le dimensioni del crop *con* padding
                })

            # === Processamento in Batch per l'Estrazione delle AU ===
            batched_aus = []
            if current_frame_faces_data_for_batch and _libreface_available:
                start_au_extraction_time = time.time()
                faces_rgb_batch_for_libreface = [data["face_rgb"] for data in current_frame_faces_data_for_batch]
                batched_aus = get_au_from_face_ndarray(faces_rgb_batch_for_libreface)
                end_au_extraction_time = time.time()
                if args.vis:
                    print(f"Frame {frame_id}: Estrazione AU per {len(faces_rgb_batch_for_libreface)} volti completata in {end_au_extraction_time - start_au_extraction_time:.4f} secondi.")
            else:
                # Se LibreFace non è disponibile o non ci sono volti, crea un placeholder
                batched_aus = [{} for _ in current_frame_faces_data_for_batch]


            # === Sottomissione dei Compiti MediaPipe al ThreadPoolExecutor ===
            if current_frame_faces_data_for_batch:
                start_mediapipe_parallel_time = time.time()

                # Associa i risultati AU ai dati del volto
                au_results_map = {
                    current_frame_faces_data_for_batch[i]["track_id"]: batched_aus[i]
                    for i in range(len(current_frame_faces_data_for_batch))
                }

                # Sottometti i compiti per il frame corrente
                for face_data in current_frame_faces_data_for_batch:
                    future = executor.submit(_process_face_mesh_for_thread, face_data["face_rgb"])
                    # Memorizza il future insieme ai dati originali del volto e le AU
                    future_tasks.append((future, face_data["track_id"], face_data["face_rgb"], 
                                         face_data["bbox"], face_data["original_bbox_dims"], 
                                         au_results_map.get(face_data["track_id"], {}))) # Aggiungi AUS

                # === Elaborazione dei Risultati dei Future completati ===
                # Non elaboriamo tutti i futuri in ogni frame, ma solo quelli completati.
                # Questo evita di bloccare il main thread.
                completed_futures_indices = []
                for i, (future, track_id, face_rgb_orig, bbox, original_bbox_dims, aus_pred) in enumerate(future_tasks):
                    if future.done():
                        try:
                            face_landmarks_result = future.result() # Ottieni il risultato del thread
                            
                            # Raccolta dati per visualizzazione e gestione clip
                            if face_landmarks_result:
                                faces_for_mediapipe_vis.append({
                                    "track_id": track_id,
                                    "face_rgb": face_rgb_orig,
                                    "face_landmarks": face_landmarks_result,
                                    "bbox": bbox, # Questo è il bbox PADDED
                                    "original_bbox_dims": original_bbox_dims, # Queste sono le dimensioni PADDED del crop
                                    "aus": aus_pred # Aggiungi le AU
                                })

                            # === Costruzione delle clip temporali di immagini (Ramo 1) ===
                            if track_id not in clip_buffer:
                                clip_buffer[track_id] = []
                                clip_start_times[track_id] = time.time()

                            clip_buffer[track_id].append(face_rgb_orig) # Aggiungi il frame corrente

                            if len(clip_buffer[track_id]) >= CLIP_LENGTH:
                                elapsed = time.time() - clip_start_times[track_id]
                                fps_clip = CLIP_LENGTH / elapsed if elapsed > 0 else 0
                                clip_data = np.stack(clip_buffer[track_id][:CLIP_LENGTH])

                                if args.mode == "save":
                                    # Usa Path per costruire percorsi cross-platform
                                    clip_img_path_npy = Path(CLIP_DIR) / f"track{track_id}_frame{frame_id}_img_clip.npy"
                                    clip_img_path_pt = Path(CLIP_DIR) / f"track{track_id}_frame{frame_id}_img_clip.pt"
                                    np.save(str(clip_img_path_npy), clip_data)
                                    torch.save(torch.tensor(clip_data).permute(0, 3, 1, 2).float() / 255.0,
                                                str(clip_img_path_pt))
                                else: # mode == "memory"
                                    clips_in_ram.append({"track_id": track_id, "clip": clip_data, "frame_id_end": frame_id})

                                clip_logs.append({"frame_id_end": frame_id, "track_id": track_id, "elapsed_time_clip_img": elapsed, "fps_clip_img": fps_clip})

                                # Avanza il buffer della clip immagine
                                clip_buffer[track_id] = clip_buffer[track_id][CLIP_STEP:]
                                clip_start_times[track_id] = time.time()

                            # === Costruzione delle finestre temporali di AU (Ramo 2) ===
                            if aus_pred: # aus_pred potrebbe essere {} se LibreFace non è disponibile
                                if track_id not in au_buffer:
                                    au_buffer[track_id] = []
                                    au_clip_start_times[track_id] = time.time()

                                au_buffer[track_id].append(aus_pred) # Aggiungi i risultati AU del frame corrente

                                if len(au_buffer[track_id]) >= AU_CLIP_LENGTH:
                                    elapsed_au = time.time() - au_clip_start_times[track_id]
                                    fps_au_clip = AU_CLIP_LENGTH / elapsed_au if elapsed_au > 0 else 0
                                    au_sequence = au_buffer[track_id][:AU_CLIP_LENGTH]

                                    # Qui potresti salvare o processare au_sequence
                                    # Esempio: print(f"Clip AU pronta per Track {track_id}: {au_sequence}")
                                    # log per le AU (puoi aggiungere un DataFrame separato se necessario)
                                    
                                    # Avanza il buffer della clip AU
                                    au_buffer[track_id] = au_buffer[track_id][AU_CLIP_STEP:]
                                    au_clip_start_times[track_id] = time.time()

                            completed_futures_indices.append(i) # Segna il future come completato

                        except Exception as exc:
                            print(f"Generata un'eccezione dal thread per Track {track_id}: {exc}")
                            completed_futures_indices.append(i) # Segna comunque come completato per rimuoverlo

                # Rimuovi i future completati dalla lista principale (dall'ultimo al primo per non sballare gli indici)
                for i in sorted(completed_futures_indices, reverse=True):
                    del future_tasks[i]
                
                if args.vis and current_frame_faces_data_for_batch:
                    end_mediapipe_parallel_time = time.time()
                    print(f"Frame {frame_id}: Elaborazione MediaPipe in parallelo per {len(current_frame_faces_data_for_batch)} volti completata in {end_mediapipe_parallel_time - start_mediapipe_parallel_time:.4f} secondi.")


            # === Visualizzazione su Frame Originale (se abilitata) ===
            if args.vis:
                # Disegna tutti i bounding box (YuNet) e i landmark (MediaPipe) raccolti
                for mp_data in faces_for_mediapipe_vis:
                    x1, y1, x2, y2 = mp_data["bbox"] # Queste sono le coordinate PADDED
                    track_id = mp_data["track_id"]
                    face_landmarks = mp_data["face_landmarks"]
                    original_w_box, original_h_box = mp_data["original_bbox_dims"] # Queste sono le dimensioni PADDED
                    aus = mp_data["aus"] # Le Action Units

                    # Disegna il bounding box (quello padded)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 1)

                    # Riposiziona i landmark sul frame originale
                    adjusted_landmarks_list = []
                    scale_x = original_w_box / CLIP_SIZE[0] # Scala dal 224x224 al crop padded
                    scale_y = original_h_box / CLIP_SIZE[1]

                    for landmark in face_landmarks.landmark:
                        x_resized = landmark.x * CLIP_SIZE[0]
                        y_resized = landmark.y * CLIP_SIZE[1]
                        x_on_original_bbox = x_resized * scale_x
                        y_on_original_bbox = y_resized * scale_y
                        
                        # Offset con l'angolo superiore sinistro del bbox padded
                        x_frame = (x1 + x_on_original_bbox) / img_w
                        y_frame = (y1 + y_on_original_bbox) / img_h

                        adjusted_landmarks_list.append(
                            landmark_pb2.NormalizedLandmark(x=x_frame, y=y_frame, z=landmark.z)
                        )
                    
                    temp_landmarks_proto = landmark_pb2.NormalizedLandmarkList(landmark=adjusted_landmarks_list)

                    # Disegna i landmark sul frame principale
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=temp_landmarks_proto,
                        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                    )
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=temp_landmarks_proto,
                        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                    )
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=temp_landmarks_proto,
                        connections=mp.solutions.face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
                    )

                    # Mostra le Action Units
                    if aus:
                        y_offset = y2 + 10 # Posiziona sotto il bbox
                        for idx, (au, val) in enumerate(aus.items()):
                            cv2.putText(frame, f"{au}:{val:.2f}", (x1, y_offset + idx * 20), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                    # Finestra di debug per il volto ritagliato con i landmark (se --show_faces è abilitato)
                    if args.show_faces:
                        debug_face_img = cv2.cvtColor(mp_data["face_rgb"].copy(), cv2.COLOR_RGB2BGR)
                        mp_drawing.draw_landmarks(image=debug_face_img, landmark_list=mp_data["face_landmarks"],
                                                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None,
                                                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                        mp_drawing.draw_landmarks(image=debug_face_img, landmark_list=mp_data["face_landmarks"],
                                                connections=mp.solutions.face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=None,
                                                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
                        mp_drawing.draw_landmarks(image=debug_face_img, landmark_list=mp_data["face_landmarks"],
                                                connections=mp.solutions.face_mesh.FACEMESH_IRISES, landmark_drawing_spec=None,
                                                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())
                        cv2.imshow(f"Cropped Face with Landmarks ID {mp_data['track_id']}", debug_face_img)
                        # Assicurati che le finestre chiuse vengano ricreate se i track_id cambiano
                        cv2.waitKey(1) # Un piccolo waitKey per consentire il refresh delle finestre secondarie


                fps = 1.0 / (time.time() - start_frame_time)
                cv2.putText(frame, f"FPS: {fps:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                cv2.imshow("Tracking & Facial Analysis", frame)

            key = cv2.waitKey(10) & 0xFF
            if key == 27: # ESC key
                break

    finally: # Questo blocco viene sempre eseguito, anche se esci dal ciclo con break o un errore
        print("Tentativo di spegnere l'executor dei thread e rilasciare risorse...")
        # Attendi il completamento di tutti i future ancora in corso prima di spegnere l'executor
        for future, _, _, _, _, _ in future_tasks:
            try:
                future.result() # Questo blocca finché il future non è completato
            except Exception as exc:
                print(f"Eccezione catturata da un thread terminato: {exc}")

        executor.shutdown(wait=True)
        print("Executor dei thread spento.")

        cap.release()
        cv2.destroyAllWindows()

        log_df = pd.DataFrame(clip_logs)

        if not log_df.empty:
            log_df.to_csv("fps_log.csv", index=False)
            print(f"Log degli FPS salvato in fps_log.csv. Totale clip immagini processate: {len(log_df)}")

            plt.figure(figsize=(10, 5))
            plt.plot(log_df["frame_id_end"], log_df["fps_clip_img"], marker='o', linestyle='-')
            plt.title("Andamento FPS medi per clip immagini salvata")
            plt.xlabel("Frame ID Fine Clip")
            plt.ylabel("FPS medi (clip)")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig("fps_plot.png")
            plt.show()
            print(f"Grafico degli FPS salvato in fps_plot.png")
        else:
            print("⚠️ Nessuna clip immagine salvata. Controlla che la webcam sia attiva, che i volti vengano rilevati o che l'input sia valido.")