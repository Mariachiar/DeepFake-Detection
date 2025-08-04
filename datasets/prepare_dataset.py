import os
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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocessing.yunet.yunet import YuNet
from preprocessing.ByteTrack.byte_tracker import BYTETracker, STrack, TrackState

import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from preprocessing.libreface.libreface_adapter import get_au_from_face_ndarray, _initialize_au_model as libreface_init_au_model
_libreface_available = True
"""try:
    from preprocessing.libreface.libreface_adapter import get_au_from_face_ndarray, _initialize_au_model as libreface_init_au_model
    print("LibreFace importato correttamente per l'estrazione delle AU.")
    _libreface_available = True
except ImportError:
    print("ATTENZIONE: libreface/libreface_adapter.py o get_au_from_face_ndarray non trovato.")
    print("Verrà utilizzata una funzione placeholder per l'estrazione delle AU (simulazione batch).")
    def get_au_from_face_ndarray(face_rgbs_batch):
        # Simulate a small delay for AU extraction
        time.sleep(0.005 * len(face_rgbs_batch))
        batch_results = []
        for _ in face_rgbs_batch:
            batch_results.append({
                "AU01": np.random.rand(), "AU02": np.random.rand(), "AU04": np.random.rand(),
                "AU06": np.random.rand(), "AU07": np.random.rand(), "AU10": np.random.rand(),
                "AU12": np.random.rand(), "AU14": np.random.rand(), "AU15": np.random.rand(),
                "AU17": np.random.rand(), "AU23": np.random.rand(), "AU24": np.random.rand()
            })
        return batch_results
    _libreface_available = False"""


CLIP_LENGTH = 8
CLIP_STEP = 4
CLIP_SIZE = (224, 224) # Dimensione per i volti ritagliati
AU_CLIP_LENGTH = 8 # Mantenuto per chiarezza, sarà uguale a CLIP_LENGTH per coerenza
AU_CLIP_STEP = 4   # Mantenuto per chiarezza, sarà uguale a CLIP_STEP per coerenza

OUTPUT_BASE_DIR = "processed_dataset" # Cartella base per l'output

backend_target_pairs = [
    [cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU],
    [cv2.dnn.DNN_BACKEND_CUDA, cv2.dnn.DNN_TARGET_CUDA]
]

parser = argparse.ArgumentParser(description="Real-time Multi-person Deepfake Preprocessing Pipeline")
parser.add_argument('--model', '-m', type=str,
                     default='C:\\Users\\maria\\Desktop\\deepfake\\preprocessing\\yunet\\face_detection_yunet_2023mar.onnx',
                     help='Percorso al modello ONNX di YuNet (per il rilevamento facciale).')
parser.add_argument('--backend_target', '-bt', type=int, default=0,
                     help='Backend e target per YuNet. 0: OpenCV-CPU (default), 1: CUDA-GPU.')
parser.add_argument('--input', '-i', type=str, default='0', # Default a '0' per webcam
                     help='Percorso al file video o alla cartella contenente i video. Se 0, usa la webcam.')
parser.add_argument('--vis', '-v', action='store_true',
                     help='Abilita la visualizzazione in tempo reale con bounding box, ID, landmark e FPS.')
parser.add_argument('--num_workers_per_face', type=int, default=os.cpu_count() or 1, # Rinominato per chiarezza
                     help='Numero di thread da usare per elaborare i singoli volti in parallelo (per MediaPipe).')
parser.add_argument('--show_faces', action='store_true',
                     help='Mostra finestre separate per ogni volto ritagliato con landmarks (solo con --vis).')
parser.add_argument('--yunet_res', type=int, default=0,
                     help='Risoluzione del lato più corto per il ridimensionamento dell\'input di YuNet (es. 320). 0 per usare la risoluzione originale del frame.')
args = parser.parse_args()

# Global counter for unique clip IDs across all videos/tracks
global_clip_index = 0

# Lista per contenere i log di tutte le clip salvate
all_clip_logs = []

# === Funzione per salvare i dati della clip ===
def save_clip_data(base_output_dir, source_name, track_id, clip_idx,
                   img_clip_data, landmarks_clip_data, aus_clip_data,
                   frame_start_id, frame_end_id):
    """
    Salva una singola clip di immagini, landmark e AU.
    Crea una struttura di directory organizzata:
    base_output_dir/
        source_name/
            track_id_X/
                clip_Y/
                    images.npy
                    landmarks.npy
                    aus.npy
    """
    global global_clip_index

    # Rimuovi estensione dal source_name se è un file
    if '.' in source_name:
        source_name = os.path.splitext(source_name)[0]
    
    # Crea il percorso per la traccia specifica
    track_output_dir = os.path.join(base_output_dir, source_name, f"track_{track_id}")
    os.makedirs(track_output_dir, exist_ok=True)

    # Crea il percorso per la clip specifica
    clip_output_dir = os.path.join(track_output_dir, f"clip_{clip_idx:05d}") # Formato a 5 cifre per ordinamento
    os.makedirs(clip_output_dir, exist_ok=True)

    # Salva le immagini della clip
    np.save(os.path.join(clip_output_dir, "images.npy"), img_clip_data)
    # Converti e salva come tensor PyTorch
    torch.save(torch.tensor(img_clip_data).permute(0, 3, 1, 2).float() / 255.0,
               os.path.join(clip_output_dir, "images.pt"))

    # Salva i landmark (converti i protocol buffer in un formato serializzabile, es. lista di liste)
    # Ogni elemento in landmarks_clip_data è un NormalizedLandmarkList.
    # Convertiamoli in liste di dizionari per la serializzazione
    serializable_landmarks = []
    for frame_landmarks in landmarks_clip_data:
        if frame_landmarks:
            frame_lm_list = []
            for lm in frame_landmarks.landmark:
                frame_lm_list.append({"x": lm.x, "y": lm.y, "z": lm.z})
            serializable_landmarks.append(frame_lm_list)
        else:
            serializable_landmarks.append([]) # Niente landmark per questo frame
    np.save(os.path.join(clip_output_dir, "landmarks.npy"), np.array(serializable_landmarks, dtype=object))
    # Puoi anche salvare come JSON per leggibilità, se preferisci
    # with open(os.path.join(clip_output_dir, "landmarks.json"), 'w') as f:
    #     json.dump(serializable_landmarks, f)


    # Salva le AU (già un dizionario per ogni frame, converti in lista di dizionari)
    np.save(os.path.join(clip_output_dir, "aus.npy"), np.array(aus_clip_data, dtype=object))
    # with open(os.path.join(clip_output_dir, "aus.json"), 'w') as f:
    #     json.dump(aus_clip_data, f)
    
    log_entry = {
        "global_clip_id": global_clip_index,
        "source_name": source_name,
        "track_id": track_id,
        "clip_idx_in_track": clip_idx,
        "clip_path": os.path.relpath(clip_output_dir, base_output_dir), # Percorso relativo alla base
        "frame_start_id": frame_start_id, # Frame ID del primo frame della clip
        "frame_end_id": frame_end_id,     # Frame ID dell'ultimo frame della clip
        "clip_length_frames": CLIP_LENGTH,
        "clip_size_pixels": CLIP_SIZE
    }
    global_clip_index += 1
    
    return log_entry

# === Funzione helper per l'elaborazione di MediaPipe (modificata per misurare il tempo) ===
# Non hai più bisogno di passare "face_mesh_detector_instance" come argomento
def _process_face_mesh_for_thread(face_rgb_input): 
    # 1. L'ISTANZA VIENE CREATA QUI, DENTRO IL THREAD
    face_mesh_detector_instance = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    thread_start_time = time.time()
    results_mesh_thread = face_mesh_detector_instance.process(face_rgb_input)
    thread_end_time = time.time()
    processing_time = thread_end_time - thread_start_time

    # 2. È buona norma chiudere l'istanza per liberare risorse
    face_mesh_detector_instance.close() 

    return results_mesh_thread.multi_face_landmarks[0] if results_mesh_thread.multi_face_landmarks else None, processing_time


if __name__ == "__main__":
    cv2.namedWindow("Tracking & Facial Analysis", cv2.WINDOW_NORMAL)
    loading_img = np.zeros((480, 640, 3), dtype=np.uint8)

    cv2.putText(loading_img, "Loading models... Please wait.", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imshow("Tracking & Facial Analysis", loading_img)
    cv2.waitKey(1)

    if _libreface_available:
        print("Forzatura dell'inizializzazione di LibreFace AU...")
        try:
            libreface_init_au_model()
            cv2.putText(loading_img, "LibreFace AU loaded.", (50, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.imshow("Tracking & Facial Analysis", loading_img)
            cv2.waitKey(1)
        except Exception as e:
            print(f"ERRORE: Inizializzazione LibreFace AU fallita: {e}")
            cv2.putText(loading_img, "LibreFace AU FAILED!", (50, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            cv2.imshow("Tracking & Facial Analysis", loading_img)
            cv2.waitKey(0)
            sys.exit(1)
    else:
        cv2.putText(loading_img, "LibreFace AU (placeholder).", (50, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 150, 255), 1)
        cv2.imshow("Tracking & Facial Analysis", loading_img)
        cv2.waitKey(1)

    backend_id, target_id = backend_target_pairs[args.backend_target]
    print(f"DEBUG: Numero massimo di thread configurati per l'executor: {args.num_workers_per_face}")

    face_detector = YuNet(
        modelPath=args.model,
        inputSize=[640, 480], # Dummy size, will be updated
        confThreshold=0.9,
        nmsThreshold=0.3,
        topK=5000,
        backendId=backend_id,
        targetId=target_id
    )
    cv2.putText(loading_img, "YuNet loaded.", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    cv2.imshow("Tracking & Facial Analysis", loading_img)
    cv2.waitKey(1)

    tracker_args = Namespace(track_thresh=0.5, match_thresh=0.8, track_buffer=30, frame_rate=15, mot20=False)
    tracker = BYTETracker(args=tracker_args, frame_rate=15)
    cv2.putText(loading_img, "ByteTrack initialized.", (50, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    cv2.imshow("Tracking & Facial Analysis", loading_img)
    cv2.waitKey(1)

    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # MediaPipe FaceMesh è inizializzato una volta per l'uso nei thread
    face_mesh_detector_global = mp_face_mesh.FaceMesh( # Rinominato per evitare confusione
        static_image_mode=False,
        max_num_faces=1, # Max faces per thread processing (each thread processes one face)
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    cv2.putText(loading_img, "MediaPipe FaceMesh loaded.", (50, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    cv2.imshow("Tracking & Facial Analysis", loading_img)
    cv2.waitKey(1)

    executor = ThreadPoolExecutor(max_workers=args.num_workers_per_face)
    
    # Futures holds tuples of (future_object, track_id, original_face_rgb, bbox, original_bbox_dims, aus_pred)
    futures = [] # Ora memorizza i dati originali per il recupero
    
    cv2.putText(loading_img, "ThreadPoolExecutor ready.", (50, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    cv2.imshow("Tracking & Facial Analysis", loading_img)
    cv2.waitKey(1)

    # --- Gestione input video/webcam ---
    video_paths = []
    source_name = "" # Default per la webcam

    if os.path.isfile(args.input):
        video_paths.append(args.input)
        source_name = os.path.basename(args.input)
    elif os.path.isdir(args.input):
        for filename in os.listdir(args.input):
            if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_paths.append(os.path.join(args.input, filename))
        if not video_paths:
            print(f"Errore: Nessun file video trovato nella cartella '{args.input}'.")
            sys.exit(1)
        source_name = os.path.basename(args.input) # Nome della cartella
    else:
        print(f"Errore: Percorso input '{args.input}' non valido. Deve essere un file video, una cartella o '0' per la webcam.")
        sys.exit(1)

    cv2.putText(loading_img, "Input source determined.", (50, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    cv2.imshow("Tracking & Facial Analysis", loading_img)
    cv2.waitKey(1000)
    cv2.destroyWindow("Tracking & Facial Analysis")

    for video_path in video_paths:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Errore: Impossibile aprire la sorgente video/webcam: {video_path}")
            continue

        current_source_name = os.path.splitext(os.path.basename(video_path))[0]
        print(f"Inizio elaborazione video: {current_source_name}")

        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_video = cap.get(cv2.CAP_PROP_FPS)
        
        # Aggiorna la dimensione di input di YuNet basandosi sulle dimensioni della webcam o sull'argomento --yunet_res
        if args.yunet_res > 0:
            if w > h:
                new_h = args.yunet_res
                new_w = int(w * (new_h / h))
            else:
                new_w = args.yunet_res
                new_h = int(h * (new_w / w))
            yunet_input_size = [new_w, new_h]
            print(f"YuNet input size set to: {yunet_input_size} (from --yunet_res {args.yunet_res})")
        else:
            yunet_input_size = [w, h]
            print(f"YuNet input size set to: {yunet_input_size} (original frame resolution)")

        face_detector.setInputSize(yunet_input_size)

        frame_id = 0
        
        # Ora i buffer delle clip sono per traccia E contengono immagini, landmark e AU
        clip_buffer_per_track = {} # {track_id: {"images": [], "landmarks": [], "aus": []}}
        clip_counter_per_track = {} # {track_id: current_clip_index_for_this_track}
        
        # Log per il tempo di ogni componente
        pipeline_logs = []
        
        # Questo verrà usato per popolare all_clip_logs alla fine
        video_clip_logs = [] 

        print(f"Acquisizione avviata per {current_source_name}. Premi 'ESC' per uscire.")

        try:
            while True:
                start_total_frame_time = time.time()
                ret, frame = cap.read()
                end_read_frame_time = time.time()
                read_frame_time = end_read_frame_time - start_total_frame_time

                frame_log = {"frame_id": frame_id + 1, "source_name": current_source_name}
                frame_log["read_frame_time"] = read_frame_time

                if not ret:
                    print(f"Fine del flusso video per {current_source_name} o errore di lettura del frame.")
                    break

                frame_id += 1

                frame_for_yunet = frame
                if args.yunet_res > 0:
                    frame_for_yunet = cv2.resize(frame, (yunet_input_size[0], yunet_input_size[1]))

                start_detection_time = time.time()
                detections = face_detector.infer(frame_for_yunet)
                end_detection_time = time.time()
                frame_log["yunet_inference_time"] = end_detection_time - start_detection_time
                # if args.vis: # Spostato all'interno del blocco di visualizzazione finale per non inquinare la console
                #     print(f"Frame {frame_id}: Tempo rilevamento YuNet: {end_detection_time - start_detection_time:.4f} secondi.")

                if args.yunet_res > 0:
                    scale_x = w / yunet_input_size[0]
                    scale_y = h / yunet_input_size[1]
                    for det in detections:
                        det[0] = det[0] * scale_x # x1
                        det[1] = det[1] * scale_y # y1
                        det[2] = det[2] * scale_x # width
                        det[3] = det[3] * scale_y # height
                        # If you were to use YuNet landmarks, they'd need scaling too:
                        # det[4], det[5] (right eye), det[6], det[7] (left eye), etc.

                faces_detected_for_tracking = [STrack(det[:4], score=det[-1]) for det in detections]

                img_h, img_w = frame.shape[:2]

                start_tracking_time = time.time()
                online_targets = tracker.update(faces_detected_for_tracking, (img_h, img_w), (img_w, img_h))
                end_tracking_time = time.time()
                frame_log["bytetrack_update_time"] = end_tracking_time - start_tracking_time
                # if args.vis: # Spostato all'interno del blocco di visualizzazione finale
                #     print(f"Frame {frame_id}: Tempo tracciamento ByteTrack: {end_tracking_time - start_tracking_time:.4f} secondi.")

                current_frame_faces_data_for_processing = [] # Dati dei volti per AU e MediaPipe
                start_face_preprocessing_time = time.time()
                for track in online_targets:
                    if not track.is_activated or track.state == TrackState.Lost:
                        continue

                    track_id = track.track_id
                    x1, y1, w_box, h_box = map(int, track.tlwh)
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(x1 + w_box, frame.shape[1])
                    y2 = min(y1 + h_box, frame.shape[0])
                    w_box = x2 - x1
                    h_box = y2 - y1

                    if w_box <= 0 or h_box <= 0:
                        continue
                    
                    face_cropped = frame[y1:y2, x1:x2]
                    if face_cropped.size == 0:
                        continue

                    face_resized = cv2.resize(face_cropped, CLIP_SIZE)
                    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)

                    current_frame_faces_data_for_processing.append({
                        "track_id": track_id,
                        "face_rgb": face_rgb, # This is the CLIP_SIZE, RGB image
                        "bbox": (x1, y1, x2, y2), # Bbox in original frame coordinates
                        "original_bbox_dims": (w_box, h_box), # Dims of bbox in original frame
                        "frame_id": frame_id # Add current frame_id for clip logging
                    })
                end_face_preprocessing_time = time.time()
                frame_log["face_preprocessing_time"] = end_face_preprocessing_time - start_face_preprocessing_time

                # --- Estrazione AU per tutti i volti del frame in batch ---
                batched_aus = []
                start_au_extraction_time = 0
                end_au_extraction_time = 0
                if current_frame_faces_data_for_processing:
                    start_au_extraction_time = time.time()
                    faces_rgb_batch_for_libreface = [data["face_rgb"] for data in current_frame_faces_data_for_processing]
                    batched_aus = get_au_from_face_ndarray(faces_rgb_batch_for_libreface)
                    end_au_extraction_time = time.time()
                    # if args.vis: # Spostato all'interno del blocco di visualizzazione finale
                    #     print(f"Frame {frame_id}: Estrazione AU per {len(faces_rgb_batch_for_libreface)} volti completata in {end_au_extraction_time - start_au_extraction_time:.4f} secondi.")
                frame_log["au_extraction_time"] = end_au_extraction_time - start_au_extraction_time

                # Associa le AU ai rispettivi dati del volto
                for i, face_data in enumerate(current_frame_faces_data_for_processing):
                    if i < len(batched_aus):
                        face_data["aus"] = batched_aus[i]
                    else:
                        face_data["aus"] = {} # O un valore di default

                # --- Elaborazione MediaPipe Face Mesh (per ogni volto in parallelo) ---
                total_mediapipe_processing_time_aggregated = 0 # Accumula il tempo di CPU di MediaPipe

                # Invia nuovi future per i volti del frame corrente
                new_futures_this_frame = []
                for face_data in current_frame_faces_data_for_processing:
                    future = executor.submit(_process_face_mesh_for_thread, face_data["face_rgb"]) 
                    # Memorizza tutti i dati necessari per la visualizzazione e il salvataggio
                    new_futures_this_frame.append((future, face_data)) # futuro e i dati del volto originali
                futures.extend(new_futures_this_frame) # Aggiungi alla lista globale dei future attivi

                # Lista temporanea per i volti elaborati in questo frame per la visualizzazione
                current_frame_processed_faces_for_vis = []
                
                # Inizializza il tempo di gestione delle clip per questo frame
                frame_log["clip_handling_time"] = 0 
                
                start_mediapipe_parallel_wall_time = time.time()

                # Raccogli i risultati dai future completati (possono provenire da frame precedenti o dal corrente)
                completed_futures_indices = []
                for i in range(len(futures) - 1, -1, -1): # Itera al contrario per una rimozione sicura
                    future, face_data_original = futures[i] # Estrai il future e i dati originali
                    
                    if future.done():
                        try:
                            # Il risultato di _process_face_mesh_for_thread è (landmarks, processing_time)
                            face_landmarks_result, mediapipe_process_time = future.result() 
                            total_mediapipe_processing_time_aggregated += mediapipe_process_time

                            # Aggiungi i landmark al dizionario del volto per il buffering/salvataggio/visualizzazione
                            face_data_original["landmarks"] = face_landmarks_result
                            
                            # Se questo future appartiene al frame corrente (per la visualizzazione immediata)
                            if face_data_original["frame_id"] >= frame_id - 1:
                                current_frame_processed_faces_for_vis.append(face_data_original)


                            # Gestione del buffer delle clip per questo volto
                            start_clip_handling_time_per_face = time.time()

                            track_id = face_data_original["track_id"]
                            
                            clip_buffer_per_track.setdefault(track_id, {
                                "images": [], "landmarks": [], "aus": [], "frame_ids": []
                            })
                            
                            clip_buffer_per_track[track_id]["images"].append(face_data_original["face_rgb"])
                            # Assicurati che i landmark siano serializzabili (es. None o NormalizedLandmarkList)
                            clip_buffer_per_track[track_id]["landmarks"].append(face_landmarks_result)
                            clip_buffer_per_track[track_id]["aus"].append(face_data_original["aus"])
                            clip_buffer_per_track[track_id]["frame_ids"].append(face_data_original["frame_id"])

                            if len(clip_buffer_per_track[track_id]["images"]) >= CLIP_LENGTH:
                                clip_idx = clip_counter_per_track.setdefault(track_id, 0)
                                
                                img_clip_data = np.stack(clip_buffer_per_track[track_id]["images"][:CLIP_LENGTH])
                                landmarks_clip = clip_buffer_per_track[track_id]["landmarks"][:CLIP_LENGTH]
                                aus_clip = clip_buffer_per_track[track_id]["aus"][:CLIP_LENGTH]
                                clip_frame_ids = clip_buffer_per_track[track_id]["frame_ids"][:CLIP_LENGTH]
                                
                                # Trova il frame ID di inizio e fine per la clip
                                start_frame_id = min(clip_frame_ids)
                                end_frame_id = max(clip_frame_ids)

                                # Salva la clip e ottieni il log entry
                                log_entry = save_clip_data(
                                    OUTPUT_BASE_DIR,
                                    current_source_name, # Passa il nome della sorgente (video file o "webcam_session")
                                    track_id,
                                    clip_idx,
                                    img_clip_data,
                                    landmarks_clip,
                                    aus_clip,
                                    start_frame_id,
                                    end_frame_id
                                )
                                video_clip_logs.append(log_entry) # Aggiungi al log di questo video
                                clip_counter_per_track[track_id] += 1

                                # Rimuovi i frame processati dal buffer
                                clip_buffer_per_track[track_id]["images"] = clip_buffer_per_track[track_id]["images"][CLIP_STEP:]
                                clip_buffer_per_track[track_id]["landmarks"] = clip_buffer_per_track[track_id]["landmarks"][CLIP_STEP:]
                                clip_buffer_per_track[track_id]["aus"] = clip_buffer_per_track[track_id]["aus"][CLIP_STEP:]
                                clip_buffer_per_track[track_id]["frame_ids"] = clip_buffer_per_track[track_id]["frame_ids"][CLIP_STEP:]

                            end_clip_handling_time_per_face = time.time()
                            frame_log["clip_handling_time"] += (end_clip_handling_time_per_face - start_clip_handling_time_per_face)

                            completed_futures_indices.append(i) # Mark for removal
                        except Exception as exc:
                            print(f"Generata un'eccezione da un thread MediaPipe/Clip Handling: {exc}")
                            completed_futures_indices.append(i) # Rimuovi il future fallito

                # Rimuovi i future completati
                for i in sorted(completed_futures_indices, reverse=True):
                    del futures[i]
                
                end_mediapipe_parallel_wall_time = time.time()
                frame_log["mediapipe_processing_time"] = total_mediapipe_processing_time_aggregated # Tempo CPU aggregato
                frame_log["mediapipe_parallel_wall_time"] = end_mediapipe_parallel_wall_time - start_mediapipe_parallel_wall_time # Tempo di parete

                end_total_frame_time = time.time()
                total_frame_time = end_total_frame_time - start_total_frame_time
                frame_log["total_pipeline_fps"] = 1.0 / total_frame_time if total_frame_time > 0 else 0
                frame_log["total_processing_time"] = total_frame_time
                
                pipeline_logs.append(frame_log)

                # === Visualizzazione ===
                if args.vis:
                    # Stampa i log di tempo solo quando la visualizzazione è attiva
                    print(f"Frame {frame_id} (FPS: {frame_log['total_pipeline_fps']:.2f}): "
                          f"YuNet: {frame_log['yunet_inference_time']:.4f}s, "
                          f"ByteTrack: {frame_log['bytetrack_update_time']:.4f}s, "
                          f"AU: {frame_log['au_extraction_time']:.4f}s, "
                          f"MediaPipe (Wall): {frame_log['mediapipe_parallel_wall_time']:.4f}s (CPU Agg: {frame_log['mediapipe_processing_time']:.4f}s)")

                    for face_data_item in current_frame_faces_data_for_processing: # Bounding box di YuNet/ByteTrack
                        x1, y1, x2, y2 = face_data_item["bbox"]
                        track_id = face_data_item["track_id"]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 1)

                    for mp_data in current_frame_processed_faces_for_vis: # Dati processati con MediaPipe e AU per la visualizzazione
                        face_landmarks = mp_data["landmarks"] # Questi sono i landmark già ottenuti
                        x1, y1, x2, y2 = mp_data["bbox"]
                        original_w_box, original_h_box = mp_data["original_bbox_dims"]
                        aus_for_vis = mp_data["aus"]
                        
                        print(f"[DEBUG] Landmark disponibili per traccia ID {mp_data['track_id']}: {face_landmarks is not None}")

                        if face_landmarks:
                            adjusted_landmarks_list = []
                            w_box_float = float(original_w_box)
                            h_box_float = float(original_h_box)
                            x1_float = float(x1)
                            y1_float = float(y1)

                            for landmark in face_landmarks.landmark:
                                pixel_x_on_original_bbox = landmark.x * w_box_float
                                pixel_y_on_original_bbox = landmark.y * h_box_float

                                pixel_x_on_original_frame = pixel_x_on_original_bbox + x1_float
                                pixel_y_on_original_frame = pixel_y_on_original_bbox + y1_float

                                adjusted_x = pixel_x_on_original_frame / img_w
                                adjusted_y = pixel_y_on_original_frame / img_h

                                adjusted_landmarks_list.append(
                                    landmark_pb2.NormalizedLandmark(x=adjusted_x, y=adjusted_y, z=landmark.z)
                                )

                            temp_landmarks_proto = landmark_pb2.NormalizedLandmarkList(landmark=adjusted_landmarks_list)

                            # Disegna le connessioni e i punti del mesh facciale sull'immagine originale
                            for landmark in adjusted_landmarks_list[:5]:
                                print(f"Landmark x: {landmark.x:.3f}, y: {landmark.y:.3f}")

                            mp_drawing.draw_landmarks(
                                image=frame,
                                landmark_list=temp_landmarks_proto,
                                connections=mp_face_mesh.FACEMESH_TESSELATION,
                                landmark_drawing_spec=None,
                                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                            )
                            mp_drawing.draw_landmarks(
                                image=frame,
                                landmark_list=temp_landmarks_proto,
                                connections=mp_face_mesh.FACEMESH_CONTOURS,
                                landmark_drawing_spec=None,
                                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                            )
                            mp_drawing.draw_landmarks(
                                image=frame,
                                landmark_list=temp_landmarks_proto,
                                connections=mp_face_mesh.FACEMESH_IRISES,
                                landmark_drawing_spec=None,
                                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
                            )

                        # Disegna le Action Units (AU)
                        if aus_for_vis:
                            y_offset = 0
                            for au_name, au_value in aus_for_vis.items():
                                au_text = f"{au_name}: {au_value:.2f}"
                                cv2.putText(frame, au_text, (x2 + 10, y1 + 20 + y_offset),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
                                y_offset += 20
                    
                    # Mostra il frame corrente e i frame ritagliati se richiesto
                    cv2.putText(frame, f"Overall FPS: {frame_log['total_pipeline_fps']:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                    cv2.imshow("Tracking & Facial Analysis", frame)
                    if args.show_faces:
                        for mp_data_for_debug in current_frame_processed_faces_for_vis:
                            debug_face_img = cv2.cvtColor(mp_data_for_debug["face_rgb"].copy(), cv2.COLOR_RGB2BGR)
                            if mp_data_for_debug["landmarks"]:
                                mp_drawing.draw_landmarks(image=debug_face_img, landmark_list=mp_data_for_debug["landmarks"],
                                                        connections=mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None,
                                                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                                mp_drawing.draw_landmarks(image=debug_face_img, landmark_list=mp_data_for_debug["landmarks"],
                                                        connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=None,
                                                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
                            cv2.imshow(f"Cropped Face ID {mp_data_for_debug['track_id']}", debug_face_img)

                else: # Modalità senza visualizzazione in tempo reale, solo status
                    status_frame = np.zeros((200, 500, 3), dtype=np.uint8)
                    cv2.putText(status_frame, f"Processing Source: {current_source_name}", (20, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(status_frame, f"Processing Frame: {frame_id}", (20, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(status_frame, f"Overall FPS: {frame_log['total_pipeline_fps']:.2f}", (20, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    cv2.putText(status_frame, "Press ESC to exit", (20, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.imshow("Pipeline Status (Press ESC to exit)", status_frame)
                    # Chiudi le finestre dei volti ritagliati se la visualizzazione è disabilitata dopo essere stata attiva
                    if cv2.getWindowProperty(f"Cropped Face ID {current_source_name}", cv2.WND_PROP_VISIBLE) > 0:
                        for window_name in [name for name in cv2.getWindowNames() if "Cropped Face ID" in name]:
                            cv2.destroyWindow(window_name)


                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    print("ESC key pressed. Exiting...")
                    break
        finally:
            cap.release() # Rilascia la cattura per il video corrente

            # Aggiungi i log di questo video al log globale
            all_clip_logs.extend(video_clip_logs)
            video_clip_logs.clear() # Resetta per il prossimo video

    # Fine dell'elaborazione di tutti i video/webcam
    print("Tentativo di spegnere l'executor dei thread...")
    # Attendi che tutti i future sottomessi completino prima di spegnere
    for future, face_data_original in futures:
        try:
            face_landmarks_result, mediapipe_process_time = future.result()

            print(f"[DEBUG] Landmark completato per track_id: {face_data_original['track_id']}, frame_id: {face_data_original['frame_id']}")
            print(f"[DEBUG] Primo punto: {face_landmarks_result.landmark[0].x:.3f}, {face_landmarks_result.landmark[0].y:.3f}")

            # Gestisci eventuali clip rimanenti che non hanno raggiunto CLIP_LENGTH
            track_id = face_data_original["track_id"]
            if track_id in clip_buffer_per_track and len(clip_buffer_per_track[track_id]["images"]) > 0:
                print(f"Gestione clip parziale per traccia {track_id}...")
                img_clip_data = np.stack(clip_buffer_per_track[track_id]["images"])
                landmarks_clip = clip_buffer_per_track[track_id]["landmarks"]
                aus_clip = clip_buffer_per_track[track_id]["aus"]
                clip_frame_ids = clip_buffer_per_track[track_id]["frame_ids"]
                
                start_frame_id = min(clip_frame_ids)
                end_frame_id = max(clip_frame_ids)

                clip_idx = clip_counter_per_track.setdefault(track_id, 0)
                log_entry = save_clip_data(
                    OUTPUT_BASE_DIR,
                    current_source_name, # Usa il nome dell'ultima sorgente elaborata
                    track_id,
                    clip_idx,
                    img_clip_data,
                    landmarks_clip,
                    aus_clip,
                    start_frame_id,
                    end_frame_id
                )
                all_clip_logs.append(log_entry) # Aggiungi al log globale

        except Exception as exc:
            print(f"Generata un'eccezione dal thread durante lo spegnimento: {exc}")

    executor.shutdown(wait=True)
    print("Executor dei thread spento.")

    cv2.destroyAllWindows()

    # Converti i log globali delle clip in DataFrame e salva
    if all_clip_logs:
        all_clip_log_df = pd.DataFrame(all_clip_logs)
        output_clips_metadata_path = os.path.join(OUTPUT_BASE_DIR, "processed_clips_metadata.csv")
        all_clip_log_df.to_csv(output_clips_metadata_path, index=False)
        print(f"Metadati delle clip processate salvati in {output_clips_metadata_path}. Totale clip: {len(all_clip_log_df)}")
    else:
        print("Nessuna clip salvata durante l'esecuzione.")


    # Converti i log di performance in DataFrame e salva
    log_df = pd.DataFrame(pipeline_logs)
    if not log_df.empty:
        output_csv_path = os.path.join(OUTPUT_BASE_DIR, f"pipeline_performance_log_{current_source_name}.csv")
        log_df.to_csv(output_csv_path, index=False)
        print(f"Log delle performance salvato in {output_csv_path}. Totale frame processati: {len(log_df)}")

        # Plotting the FPS
        plt.figure(figsize=(12, 6))
        plt.plot(log_df["frame_id"], log_df["total_pipeline_fps"], marker='o', linestyle='-', color='blue', label='Total Pipeline FPS (End-to-End)')
        plt.title(f"FPS Totale della Pipeline (End-to-End) nel Tempo - {current_source_name}")
        plt.xlabel("ID Frame")
        plt.ylabel("FPS")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_BASE_DIR, f"total_pipeline_fps_{current_source_name}.png"))
        plt.show()

        # Plotting Time Spent Per Component
        plt.figure(figsize=(12, 8))
        time_cols = ["read_frame_time", "yunet_inference_time", "bytetrack_update_time",
                     "face_preprocessing_time", "au_extraction_time", "mediapipe_processing_time",
                     "mediapipe_parallel_wall_time", "clip_handling_time"]

        actual_time_cols_to_plot = [col for col in time_cols if col in log_df.columns]

        for col in actual_time_cols_to_plot:
            plt.plot(log_df["frame_id"], log_df[col], linestyle='-',
                     label=col.replace("_", " ").replace("mediapipe parallel wall time", "MediaPipe Parallel Wall-Clock Time").replace("au", "AU").title())

        plt.title(f"Tempo di Esecuzione per Componente (Secondi) - {current_source_name}")
        plt.xlabel("ID Frame")
        plt.ylabel("Tempo (Secondi)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_BASE_DIR, f"time_spent_per_component_{current_source_name}.png"))
        plt.show()

        print(f"Grafici delle performance salvati per {current_source_name}.")
    else:
        print("⚠️ Nessun dato di performance registrato. Assicurati che il pipeline abbia processato dei frame.")