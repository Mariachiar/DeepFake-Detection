import os
import cv2
import numpy as np
import time
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import torch

from argparse import Namespace

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Importations specific to YuNet (face detection) and ByteTrack (multi-object tracking)
from preprocessing.yunet.yunet import YuNet
from preprocessing.ByteTrack.byte_tracker import BYTETracker, STrack
from preprocessing.ByteTrack.byte_tracker import TrackState

# Importations for MediaPipe (facial landmark analysis)
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
# Gestione dell'importazione di LibreFace con un placeholder, come nella versione precedente
try:
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
    _libreface_available = False

# === Global Pipeline Configurations ===
CLIP_LENGTH = 8
CLIP_STEP = 4
CLIP_SIZE = (224, 224)
AU_CLIP_LENGTH = 8
AU_CLIP_STEP = 4

CLIP_DIR = "clips_sequential" # Modifica qui per non sovrascrivere i clip della versione parallela
os.makedirs(CLIP_DIR, exist_ok=True)

input_size = [320, 320] # Questo sarà aggiornato con la risoluzione reale del frame
backend_target_pairs = [
    [cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU],
    [cv2.dnn.DNN_BACKEND_CUDA, cv2.dnn.DNN_TARGET_CUDA]
]

def main_sequential(): # Rinomina la funzione main in main_sequential
    """
    Main function to run the real-time multi-person Deepfake preprocessing pipeline (SEQUENTIAL).
    Handles initialization, video processing, and logging.
    """
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    parser = argparse.ArgumentParser(description="Real-time Multi-person Deepfake Preprocessing Pipeline (Sequential)")
    parser.add_argument('--model', '-m', type=str,
                        default='C:\\Users\\maria\\Desktop\\deepfake\\preprocessing\\yunet\\face_detection_yunet_2023mar.onnx',
                        help='Path to the YuNet ONNX model (for face detection).')
    parser.add_argument('--backend_target', '-bt', type=int, default=0,
                        help='Backend and target for YuNet. 0: OpenCV-CPU (default), 1: CUDA-GPU.')
    parser.add_argument('--mode', type=str, choices=['save', 'memory'], default='save',
                        help='Clip handling mode: "save" to disk, "memory" to keep in RAM.')
    parser.add_argument('--vis', '-v', action='store_true',
                        help='Enable real-time visualization with bounding boxes, IDs, landmarks, and FPS.')
    parser.add_argument('--yunet_res', type=int, default=0,
                         help='Risoluzione del lato più corto per il ridimensionamento dell\'input di YuNet (es. 320). 0 per usare la risoluzione originale del frame.')
    parser.add_argument('--show_faces', action='store_true',
                        help='Mostra finestre separate per ogni volto ritagliato con landmarks.')
    args = parser.parse_args()

    # Loading screen (rimosso per brevità, ma puoi reinserirlo se vuoi)
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

    face_detector = YuNet(
        modelPath=args.model,
        inputSize=input_size, # Dummy size, will be updated
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

    # Initialize MediaPipe Face Mesh once
    face_mesh_detector = mp_face_mesh.FaceMesh( # Rinominato per evitare conflitto con modulo mp_face_mesh
        static_image_mode=False,
        max_num_faces=1, # MediaPipe processes one face at a time (sequential)
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    cv2.putText(loading_img, "MediaPipe FaceMesh loaded.", (50, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    cv2.imshow("Tracking & Facial Analysis", loading_img)
    cv2.waitKey(1)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Errore: Impossibile aprire la webcam. Assicurati che sia connessa e disponibile.")
        cv2.putText(loading_img, "WEBCAM FAILED!", (50, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        cv2.imshow("Tracking & Facial Analysis", loading_img)
        cv2.waitKey(0)
        sys.exit()

    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Update YuNet input size
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

    cv2.putText(loading_img, "Webcam initialized.", (50, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    cv2.imshow("Tracking & Facial Analysis", loading_img)
    cv2.waitKey(1000)

    cv2.destroyWindow("Tracking & Facial Analysis") # Chiude la schermata di caricamento

    frame_id = 0
    clip_buffer = {}
    au_buffer = {}
    clip_start_times = {}
    au_clip_start_times = {}

    pipeline_logs = [] # List to store frame logs

    clips_in_ram = [] # For 'memory' mode, to hold clips

    print(f"Acquisizione video avviata (versione SEQUENZIALE). Premi 'ESC' per uscire.")

    try:
        while True:
            # Inizializzazione delle variabili di tempo per il frame corrente
            time_read_frame = 0
            time_yunet_infer = 0
            time_bytetrack_update = 0
            time_face_preprocessing = 0 # Tempo per cropping, resizing, color conversion
            time_au_extraction = 0
            time_mediapipe_processing = 0 # Tempo CPU totale di MediaPipe (somma per tutti i volti)
            time_clip_handling = 0 # Tempo per la logica di buffer e salvataggio/copia dei clip
            time_drawing_on_frame_overall = 0 # Tempo totale per tutte le operazioni di visualizzazione OpenCV

            start_total_pipeline_time = time.time() # Start for total pipeline FPS

            # --- Misurazione: Read Frame ---
            start_read_frame_time = time.time()
            ret, frame = cap.read()
            end_read_frame_time = time.time()
            time_read_frame = end_read_frame_time - start_read_frame_time

            if not ret:
                print("Fine del flusso video o errore di lettura del frame.")
                break

            frame_id += 1

            frame_for_yunet = frame
            if args.yunet_res > 0:
                frame_for_yunet = cv2.resize(frame, (yunet_input_size[0], yunet_input_size[1]))

            # --- Misurazione: YuNet Inference ---
            start_detection_time = time.time()
            detections = face_detector.infer(frame_for_yunet)
            end_detection_time = time.time()
            time_yunet_infer = end_detection_time - start_detection_time

            # Scala le bounding box se YuNet ha processato un'immagine ridimensionata
            if args.yunet_res > 0:
                scale_x = w / yunet_input_size[0]
                scale_y = h / yunet_input_size[1]
                for det in detections:
                    det[0] = det[0] * scale_x
                    det[1] = det[1] * scale_y
                    det[2] = det[2] * scale_x
                    det[3] = det[3] * scale_y

            faces_detected_for_tracking = [STrack(det[:4], score=det[-1]) for det in detections]
            img_h, img_w = frame.shape[:2]

            # --- Misurazione: ByteTrack Update ---
            start_tracking_time = time.time()
            online_targets = tracker.update(faces_detected_for_tracking, (img_h, img_w), (img_w, img_h))
            end_tracking_time = time.time()
            time_bytetrack_update = end_tracking_time - start_tracking_time

            current_frame_faces_data = [] # Dati dei volti per il frame corrente
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

                face_cropped = frame[y1:y2, x1:x2]
                if face_cropped.size == 0 or w_box <= 0 or h_box <= 0:
                    continue

                face_resized = cv2.resize(face_cropped, CLIP_SIZE)
                face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)

                current_frame_faces_data.append({
                    "track_id": track_id,
                    "face_rgb": face_rgb,
                    "bbox": (x1, y1, x2, y2),
                    "original_bbox_dims": (w_box, h_box)
                })
            end_face_preprocessing_time = time.time()
            time_face_preprocessing = end_face_preprocessing_time - start_face_preprocessing_time
            num_faces_processed_this_frame = len(current_frame_faces_data) # Nuovo nome per coerenza

            # --- Misurazione: AU Extraction ---
            batched_aus = []
            start_au_extraction_time = time.time()
            if current_frame_faces_data:
                faces_rgb_batch_for_libreface = [data["face_rgb"] for data in current_frame_faces_data]
                batched_aus = get_au_from_face_ndarray(faces_rgb_batch_for_libreface)
            end_au_extraction_time = time.time()
            time_au_extraction = end_au_extraction_time - start_au_extraction_time

            # --- Misurazione: MediaPipe Face Mesh (SEQUENZIALE) e Clip Handling ---
            faces_for_mediapipe_vis = []
            total_mediapipe_cpu_work_time = 0 # Accumula il tempo di processing di MediaPipe per ogni volto
            
            for i, face_data in enumerate(current_frame_faces_data):
                track_id = face_data["track_id"]
                face_rgb_input = face_data["face_rgb"]
                bbox = face_data["bbox"]
                original_bbox_dims = face_data["original_bbox_dims"]
                aus_pred = batched_aus[i] if batched_aus and i < len(batched_aus) else None # Get corresponding AU results

                # MediaPipe Processing per singolo volto
                mp_start_time = time.time()
                results_mesh = face_mesh_detector.process(face_rgb_input)
                mp_end_time = time.time()
                single_face_mediapipe_time = mp_end_time - mp_start_time
                total_mediapipe_cpu_work_time += single_face_mediapipe_time # Accumula il tempo di CPU

                if results_mesh.multi_face_landmarks:
                    faces_for_mediapipe_vis.append({
                        "track_id": track_id,
                        "face_rgb": face_rgb_input,
                        "face_landmarks": results_mesh.multi_face_landmarks[0],
                        "bbox": bbox,
                        "original_bbox_dims": original_bbox_dims
                    })
                
                # Clip handling logic per ogni volto
                start_clip_handling_local = time.time()
                if track_id not in clip_buffer:
                    clip_buffer[track_id] = []
                    clip_start_times[track_id] = time.time()
                clip_buffer[track_id].append(face_rgb_input)

                if len(clip_buffer[track_id]) >= CLIP_LENGTH:
                    clip_data = np.stack(clip_buffer[track_id][:CLIP_LENGTH])

                    if args.mode == "save":
                        np.save(f"{CLIP_DIR}/track{track_id}_frame{frame_id}_img_clip.npy", clip_data)
                        torch.save(torch.tensor(clip_data).permute(0, 3, 1, 2).float() / 255.0,
                                    f"{CLIP_DIR}/track{track_id}_frame{frame_id}_img_clip.pt")
                    else:
                        clips_in_ram.append({"track_id": track_id, "clip": clip_data, "frame_id_end": frame_id})

                    clip_buffer[track_id] = clip_buffer[track_id][CLIP_STEP:]
                    clip_start_times[track_id] = time.time()
                
                # AU clip handling (analogo al clip handling)
                if aus_pred:
                    if track_id not in au_buffer:
                        au_buffer[track_id] = []
                        au_clip_start_times[track_id] = time.time()
                    au_buffer[track_id].append(aus_pred)

                    if len(au_buffer[track_id]) >= AU_CLIP_LENGTH:
                        au_sequence = au_buffer[track_id][:AU_CLIP_LENGTH]
                        # Salva/processa au_sequence qui se necessario
                        au_buffer[track_id] = au_buffer[track_id][AU_CLIP_STEP:]
                        au_clip_start_times[track_id] = time.time()
                end_clip_handling_local = time.time()
                # Accumula il tempo di gestione del clip per questo volto nel tempo totale del frame
                time_clip_handling += (end_clip_handling_local - start_clip_handling_local)

            time_mediapipe_processing = total_mediapipe_cpu_work_time # Tempo totale CPU di MediaPipe in questo frame

            # --- Misurazione: Disegno (se vis abilitata) ---
            time_drawing_on_frame_overall = 0
            time_mediapipe_vis_drawing = 0 # Tempo solo per il disegno dei landmark di MediaPipe

            start_drawing_on_frame_overall_time = time.time()
            if args.vis:
                for face_data_item in current_frame_faces_data:
                    x1, y1, x2, y2 = face_data_item["bbox"]
                    track_id = face_data_item["track_id"]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 1)

                start_mediapipe_vis_drawing_time = time.time()
                for mp_data in faces_for_mediapipe_vis:
                    face_landmarks = mp_data["face_landmarks"]
                    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = mp_data["bbox"]
                    original_w_box, original_h_box = mp_data["original_bbox_dims"]

                    adjusted_landmarks_list = []
                    w_box_float = float(original_w_box)
                    h_box_float = float(original_h_box)
                    x1_float = float(bbox_x1)
                    y1_float = float(bbox_y1)

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
                    if args.show_faces:
                        debug_face_img = cv2.cvtColor(mp_data["face_rgb"].copy(), cv2.COLOR_RGB2BGR)
                        mp_drawing.draw_landmarks(image=debug_face_img, landmark_list=mp_data["face_landmarks"],
                                                    connections=mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=None,
                                                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                        mp_drawing.draw_landmarks(image=debug_face_img, landmark_list=mp_data["face_landmarks"],
                                                    connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=None,
                                                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
                        cv2.imshow(f"Cropped Face with Landmarks ID {mp_data['track_id']}", debug_face_img)

                end_mediapipe_vis_drawing_time = time.time()
                time_mediapipe_vis_drawing = end_mediapipe_vis_drawing_time - start_mediapipe_vis_drawing_time

                cv2.putText(frame, f"FPS: {1.0 / (time.time() - start_total_pipeline_time):.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                cv2.imshow("Tracking & Facial Analysis", frame)
            else:
                status_frame = np.zeros((200, 500, 3), dtype=np.uint8)
                cv2.putText(status_frame, f"Processing Frame: {frame_id}", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(status_frame, f"Total FPS: {1.0 / (time.time() - start_total_pipeline_time):.2f}", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(status_frame, "Press ESC to exit", (20, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imshow("Pipeline Status (Press ESC to exit)", status_frame)
            end_drawing_on_frame_overall_time = time.time()
            time_drawing_on_frame_overall = end_drawing_on_frame_overall_time - start_drawing_on_frame_overall_time
            
            key = cv2.waitKey(10) & 0xFF
            if key == 27:
                print("ESC key pressed. Exiting...")
                break

            # === Calcolo e Registrazione FPS Totale per Frame ===
            end_total_pipeline_time = time.time()
            total_processing_time = end_total_pipeline_time - start_total_pipeline_time
            total_pipeline_fps = 1.0 / total_processing_time if total_processing_time > 0 else 0

            # Registrazione delle metriche nel log del frame
            frame_log_entry = {
                "frame_id": frame_id,
                "read_frame_time": time_read_frame,
                "yunet_inference_time": time_yunet_infer,
                "bytetrack_update_time": time_bytetrack_update,
                "face_preprocessing_time": time_face_preprocessing,
                "au_extraction_time": time_au_extraction,
                "mediapipe_processing_time": time_mediapipe_processing, # Tempo CPU totale di MediaPipe
                "mediapipe_parallel_wall_time": 0.0, # Sempre 0.0 per la versione sequenziale
                "clip_handling_time": time_clip_handling,
                "drawing_on_frame_overall_time": time_drawing_on_frame_overall,
                "total_pipeline_fps": total_pipeline_fps,
                "total_processing_time": total_processing_time, # Tempo end-to-end per il frame
                "num_faces_processed": num_faces_processed_this_frame
            }
            pipeline_logs.append(frame_log_entry)

    finally:
        cap.release()
        cv2.destroyAllWindows()

        log_df = pd.DataFrame(pipeline_logs)

        if not log_df.empty:
            output_csv_path = "pipeline_performance_log_sequential.csv" # Nome file CSV modificato
            log_df.to_csv(output_csv_path, index=False)
            print(f"Log dettagliato delle performance salvato in {output_csv_path}. Frame processati: {len(log_df)}")

            # Plotting FPS
            plt.figure(figsize=(12, 6))
            plt.plot(log_df["frame_id"], log_df["total_pipeline_fps"], marker='o', linestyle='-', color='red', label='Total Pipeline FPS (End-to-End)')
            plt.title("FPS Totale della Pipeline (End-to-End) nel Tempo - Sequenziale")
            plt.xlabel("ID Frame")
            plt.ylabel("FPS")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig("total_pipeline_fps_sequential.png")
            plt.show()

            # Plotting Time Spent Per Component
            plt.figure(figsize=(12, 8))
            # Le colonne per il plot devono corrispondere ai nomi nel CSV
            time_cols = [
                "read_frame_time",
                "yunet_inference_time",
                "bytetrack_update_time",
                "face_preprocessing_time",
                "au_extraction_time",
                "mediapipe_processing_time", # Qui c'è il tempo totale CPU per MediaPipe
                "clip_handling_time",
                "drawing_on_frame_overall_time"
            ]

            actual_time_cols_to_plot = [col for col in time_cols if col in log_df.columns]

            for col in actual_time_cols_to_plot:
                plt.plot(log_df["frame_id"], log_df[col], linestyle='-',
                         label=col.replace("_", " ").replace("mediapipe", "MediaPipe").replace("au", "AU").title())

            plt.title("Tempo di Esecuzione per Componente (Secondi) - Sequenziale")
            plt.xlabel("ID Frame")
            plt.ylabel("Tempo (Secondi)")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig("time_spent_per_component_sequential.png")
            plt.show()

            print(f"Grafici delle performance salvati.")
        else:
            print("⚠️ Nessun dato di performance registrato. Assicurati che il pipeline abbia processato dei frame.")

if __name__ == "__main__":
    main_sequential()