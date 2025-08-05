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
import threading
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocessing.yunet.yunet import YuNet
from preprocessing.ByteTrack.byte_tracker import BYTETracker, STrack, TrackState
from preprocessing.ByteTrack.basetrack import BaseTrack

import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import atexit

from tqdm import tqdm


from concurrent.futures import ThreadPoolExecutor

# Flag di uscita

cleanup_called = threading.Event()

import time

esc_pressed = threading.Event()

def esc_listener():
    try:
        import termios
        import tty
        import select

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)

        try:
            tty.setcbreak(fd)
            while not esc_pressed.is_set():
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.read(1)
                    if key == '\x1b':  # ESC
                        print("\n[INFO] ESC premuto da tastiera.")
                        esc_pressed.set()
                        break
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    except (ImportError, AttributeError, OSError):
        # Windows fallback: non-blocking keyboard listener
        try:
            import msvcrt
            while not esc_pressed.is_set():
                if msvcrt.kbhit() and msvcrt.getch() == b'\x1b':
                    print("\n[INFO] ESC premuto da tastiera (Windows).")
                    esc_pressed.set()
                    break
                time.sleep(0.1)
        except ImportError:
            print("[WARN] Nessun metodo disponibile per ascoltare ESC.")

# --- LibreFace (AU Extraction) ---
try:
    from preprocessing.libreface.libreface_adapter import get_au_from_face_ndarray, _initialize_au_model as libreface_init_au_model
    print("LibreFace imported successfully for AU extraction.")
    _libreface_available = True
except ImportError:
    print("WARNING: LibreFace not found.")
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))
    print("Using a placeholder function for AU extraction (batch simulation).")
    def get_au_from_face_ndarray(face_rgbs_batch):
        time.sleep(0.005 * len(face_rgbs_batch))
        return [{
            "AU01": np.random.rand(), "AU02": np.random.rand(), "AU04": np.random.rand(),
            "AU06": np.random.rand(), "AU07": np.random.rand(), "AU10": np.random.rand(),
            "AU12": np.random.rand(), "AU14": np.random.rand(), "AU15": np.random.rand(),
            "AU17": np.random.rand(), "AU23": np.random.rand(), "AU24": np.random.rand()
        } for _ in face_rgbs_batch]
    _libreface_available = False

# --- Constants ---
CLIP_LENGTH = 8
CLIP_STEP = 4
CLIP_SIZE = (224, 224)
AU_CLIP_LENGTH = 8
AU_CLIP_STEP = 4
LAND_CLIP_LENGTH = 8
LAND_CLIP_STEP = 4
OUTPUT_BASE_DIR = "./datasets/processed_dataset"
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

backend_target_pairs = [
    [cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU],
    [cv2.dnn.DNN_BACKEND_CUDA, cv2.dnn.DNN_TARGET_CUDA]
]

# --- Argument Parser ---
parser = argparse.ArgumentParser(description="Real-time Multi-person Deepfake Preprocessing Pipeline")
parser.add_argument('--model', '-m', type=str, default=os.path.join('preprocessing', 'yunet', 'face_detection_yunet_2023mar.onnx'), help='Path to the YuNet ONNX model for face detection.')
parser.add_argument('--backend_target', '-bt', type=int, default=0, help='Backend and target for YuNet. 0: OpenCV-CPU (default), 1: CUDA-GPU.')
parser.add_argument('--mode', type=str, choices=['save', 'memory'], default='save', help='Clip handling mode: "save" to disk, "memory" to keep in RAM.')
parser.add_argument('--vis', '-v', action='store_true', help='Enable real-time visualization with bounding boxes, IDs, landmarks, and FPS.')
parser.add_argument('--num_workers_per_frame', type=int, default=os.cpu_count() or 1, help='Number of threads for parallel face processing (MediaPipe).')
parser.add_argument('--show_faces', action='store_true', help='Show separate windows for each cropped face with landmarks.')
parser.add_argument('--yunet_res', type=int, default=0, help='Shortest side resolution for YuNet input resizing (e.g., 320). 0 for original frame resolution.')
parser.add_argument('--input', '-i', type=str, default='0', help="Path to video file or folder. '0' for webcam.")
parser.add_argument('--headless', action='store_true', help='Disable all visualizations and plots (for headless/Docker environments).')
parser.add_argument('--output', type=str, default=None, help="Path to save output video. If not set, no video will be saved.")
args = parser.parse_args()

# --- Global Data Structures ---
global_clip_index = 0
all_clip_logs = [] # Aggregates clip metadata from all videos

# --- Thread-local storage for MediaPipe instances ---
thread_local_storage = threading.local()


def _get_face_mesh_detector():
    """Gets or creates a FaceMesh instance for the current thread."""
    if not hasattr(thread_local_storage, 'face_mesh_detector'):
        #if not args.headless:
        print(f"Initializing MediaPipe FaceMesh for thread {threading.get_ident()}...")
        
        # --- Modifica qui ---
        # Ripristiniamo le impostazioni originali che sono più robuste
        thread_local_storage.face_mesh_detector = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,         # <-- RIPORTATO A False
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5      # <-- REINSERITO
        )
        # --- Fine modifica ---
        
    return thread_local_storage.face_mesh_detector

def _process_face_mesh_for_thread(face_rgb_input):
    face_mesh_detector_instance = _get_face_mesh_detector()

    start_time = time.time()
    results_mesh = face_mesh_detector_instance.process(face_rgb_input)
    processing_time = time.time() - start_time

    if not results_mesh.multi_face_landmarks:
        print("[DEBUG] ⚠️ MediaPipe non ha rilevato nessun landmark per questa faccia.")
    #else:
        #print("[DEBUG] ✅ Landmark trovati.")

    landmarks = results_mesh.multi_face_landmarks[0] if results_mesh.multi_face_landmarks else None
    return landmarks, processing_time


def save_clip_data(base_output_dir, source_name, track_id, clip_idx,
                   img_clip_data, landmarks_clip_data, aus_clip_data,
                   frame_start_id, frame_end_id,
                   full_video_path=None, input_base=None):
    """Saves a single clip of images, landmarks, and AUs to disk, preserving subfolder structure."""
    global global_clip_index

    # Determina il path di output: struttura replicata oppure fallback
    if full_video_path and input_base:
        relative_path = os.path.relpath(full_video_path, input_base)
        relative_path_no_ext = os.path.splitext(relative_path)[0]
        track_output_dir = os.path.join(base_output_dir, relative_path_no_ext, f"track_{track_id}")
    else:
        source_name_no_ext = os.path.splitext(source_name)[0]
        track_output_dir = os.path.join(base_output_dir, source_name_no_ext, f"track_{track_id}")

    clip_output_dir = os.path.join(track_output_dir, f"clip_{clip_idx:05d}")
    os.makedirs(clip_output_dir, exist_ok=True)

    # --- Salvataggio immagini ---
    np.save(os.path.join(clip_output_dir, "images.npy"), img_clip_data)
    torch.save(
        torch.tensor(img_clip_data, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0,
        os.path.join(clip_output_dir, "images.pt")
    )

    # --- Salvataggio landmarks ---
    serializable_landmarks = []
    for frame_landmarks in landmarks_clip_data:
        if frame_landmarks:
            serializable_landmarks.append(
                [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in frame_landmarks.landmark]
            )
        else:
            serializable_landmarks.append([])  # Frame senza landmark

    np.save(os.path.join(clip_output_dir, "landmarks.npy"), np.array(serializable_landmarks, dtype=object))

    # --- Salvataggio AUs ---
    np.save(os.path.join(clip_output_dir, "aus.npy"), np.array(aus_clip_data, dtype=object))

    # --- Log ---
    log_entry = {
        "global_clip_id": global_clip_index,
        "source_name": os.path.splitext(source_name)[0],
        "track_id": track_id,
        "clip_idx_in_track": clip_idx,
        "clip_path": os.path.relpath(clip_output_dir, base_output_dir),
        "frame_start_id": frame_start_id,
        "frame_end_id": frame_end_id,
        "clip_length_frames": CLIP_LENGTH,
        "clip_size_pixels": f"{CLIP_SIZE[0]}x{CLIP_SIZE[1]}"
    }

    global_clip_index += 1
    return log_entry


def detect_and_track(frame, face_detector, tracker, yunet_input_size, frame_log):
    """Performs face detection (YuNet) and tracking (ByteTrack)."""
    h, w = frame.shape[:2]
    frame_for_yunet = frame
    if yunet_input_size != [w, h]:
        frame_for_yunet = cv2.resize(frame, (yunet_input_size[0], yunet_input_size[1]))

    start_time = time.time()
    detections = face_detector.infer(frame_for_yunet)
    frame_log["yunet_inference_time"] = time.time() - start_time

    if yunet_input_size != [w, h]:
        scale_x, scale_y = w / yunet_input_size[0], h / yunet_input_size[1]
        for det in detections:
            det[0], det[2] = det[0] * scale_x, det[2] * scale_x
            det[1], det[3] = det[1] * scale_y, det[3] * scale_y
    
    faces_detected_for_tracking = [STrack(det[:4], score=det[-1]) for det in detections]
    start_time = time.time()
    online_targets = tracker.update(faces_detected_for_tracking, (h, w), (w, h))
    frame_log["bytetrack_update_time"] = time.time() - start_time

    return online_targets, h, w

def preprocess_and_extract_features(frame, online_targets, frame_log):
    """Crops faces, resizes them, and extracts Action Units (AUs)."""
    start_time = time.time()
    faces_data = []
    for track in online_targets:
        if not track.is_activated or track.state == TrackState.Lost:
            continue
        
        x1, y1, w_box, h_box = map(int, track.tlwh)
        x2, y2 = x1 + w_box, y1 + h_box
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        
        face_cropped = frame[y1:y2, x1:x2]
        if face_cropped.size == 0:
            continue
        
        face_resized = cv2.resize(face_cropped, CLIP_SIZE)
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        
        faces_data.append({
            "track_id": track.track_id,
            "face_rgb": face_rgb,
            "bbox": (x1, y1, x2, y2),
            "original_bbox_dims": (w_box, h_box)
        })
    
    frame_log["face_preprocessing_time"] = time.time() - start_time
    au_results_map = {}
    if faces_data:
        start_au_time = time.time()
        faces_rgb_batch = [data["face_rgb"] for data in faces_data]
        batched_aus = get_au_from_face_ndarray(faces_rgb_batch)
        for i, data in enumerate(faces_data):
            au_results_map[data["track_id"]] = batched_aus[i]
        frame_log["au_extraction_time"] = time.time() - start_au_time
    else:
        frame_log["au_extraction_time"] = 0.0

    frame_log["num_faces"] = len(faces_data)
    return faces_data, au_results_map

def submit_tasks_to_executor(faces_data, au_results_map, executor):
    """Submits FaceMesh analysis tasks to the thread pool executor."""
    new_futures = []
    for face_data in faces_data:
        future = executor.submit(_process_face_mesh_for_thread, face_data["face_rgb"])
        new_futures.append({
            "future": future,
            "track_id": face_data["track_id"],
            "face_rgb": face_data["face_rgb"],
            "bbox": face_data["bbox"],
            "original_bbox_dims": face_data["original_bbox_dims"],
            "aus_pred": au_results_map.get(face_data["track_id"])
        })
    return new_futures

def collect_completed_futures(active_futures):
    """Checks for completed futures, collects their results, and returns them."""
    completed_results, remaining_futures = [], []
    for task in active_futures:
        if task["future"].done():
            try:
                landmarks, process_time = task["future"].result()
                if landmarks:
                    task["face_landmarks"] = landmarks
                    task["processing_time"] = process_time
                    completed_results.append(task)
            except Exception as e:
                print(f"ERROR: Task for track_id {task.get('track_id')} failed: {e}")
        else:
            remaining_futures.append(task)
    return completed_results, remaining_futures

def handle_clip_buffers(result, clip_buffer, au_buffer, land_buffer, frame_id, clips_in_ram,
                        video_clip_logs, track_clip_counters, current_source_name):
    """Manages buffering and saving/storing of clips for each tracked face."""
    track_id = result["track_id"]
    for buffer in [clip_buffer, au_buffer, land_buffer]:
        buffer.setdefault(track_id, [])

    clip_buffer[track_id].append(result["face_rgb"])
    au_buffer[track_id].append(result["aus_pred"])
    land_buffer[track_id].append(result["face_landmarks"])

    if (len(clip_buffer[track_id]) >= CLIP_LENGTH and
        len(au_buffer[track_id]) >= AU_CLIP_LENGTH and
        len(land_buffer[track_id]) >= LAND_CLIP_LENGTH):

        clip_data = np.stack(clip_buffer[track_id][:CLIP_LENGTH])
        au_sequence = au_buffer[track_id][:AU_CLIP_LENGTH]
        land_sequence = land_buffer[track_id][:LAND_CLIP_LENGTH]

        if args.mode == "memory":
            images_tensor = torch.tensor(clip_data, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
            # Further processing for AU/Landmark tensors can be done here if needed
            clips_in_ram.append({"track_id": track_id, "images": images_tensor})

        elif args.mode == "save":
            track_clip_counters.setdefault(track_id, 0)
            clip_idx = track_clip_counters[track_id]
            
            log_entry = save_clip_data(
                OUTPUT_BASE_DIR, current_source_name, track_id, clip_idx,
                clip_data, land_sequence, au_sequence,
                frame_id - CLIP_LENGTH + 1, frame_id,
                full_video_path=video_path,  # nuovo argomento
                input_base=args.input         # nuova base da cui calcolare il path relativo
            )

            video_clip_logs.append(log_entry)
            track_clip_counters[track_id] += 1

        # Slide the buffers
        clip_buffer[track_id] = clip_buffer[track_id][CLIP_STEP:]
        au_buffer[track_id] = au_buffer[track_id][AU_CLIP_STEP:]
        land_buffer[track_id] = land_buffer[track_id][LAND_CLIP_STEP:]

def draw_visualizations(frame, tracked_faces, mesh_results, img_w, img_h, frame_log, frame_id):
    """Annota bounding box, ID e landmark sul frame, compatibile con modalità headless."""

    # --- Disegna i bounding box e ID ---
    for face in tracked_faces:
        x1, y1, x2, y2 = face["bbox"]
        track_id = face["track_id"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 1)

    for mp_data in mesh_results:
        face_landmarks = mp_data["face_landmarks"]
        if face_landmarks is None:
            continue

        bbox_x1, bbox_y1, _, _ = mp_data["bbox"]
        w_box, h_box = mp_data["original_bbox_dims"]
        x1f, y1f = float(bbox_x1), float(bbox_y1)

        adjusted_landmarks = []
        for lm in face_landmarks.landmark:
            px = lm.x * w_box + x1f
            py = lm.y * h_box + y1f
            adjusted_landmarks.append(landmark_pb2.NormalizedLandmark(
                x=px / img_w, y=py / img_h, z=lm.z))

        temp_landmarks_proto = landmark_pb2.NormalizedLandmarkList(landmark=adjusted_landmarks)

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

        # Visualizza faccia croppata solo se in modalità GUI
        if args.vis and args.show_faces and not args.headless:
            debug_face_img = cv2.cvtColor(mp_data["face_rgb"].copy(), cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image=debug_face_img,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
            mp_drawing.draw_landmarks(
                image=debug_face_img,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
            )
            cv2.imshow(f"Cropped Face with Landmarks ID {mp_data['track_id']}", debug_face_img)
            cv2.waitKey(1)

    # --- Visualizzazione finestra generale (solo se non headless) ---
    if not args.headless and 'frame' in locals():
        if args.vis:
            cv2.putText(frame, f"FPS: {frame_log['total_pipeline_fps']:.2f}", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.imshow("Tracking & Facial Analysis", frame)
            cv2.waitKey(1)
        else:
            status_frame = np.zeros((200, 500, 3), dtype=np.uint8)
            cv2.putText(status_frame, f"Processing Frame: {frame_id}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(status_frame, f"Total FPS: {frame_log['total_pipeline_fps']:.2f}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(status_frame, "Press ESC to exit", (20, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Pipeline Status (Press ESC to exit)", status_frame)


def cleanup():
    if cleanup_called.is_set():
        return
    cleanup_called.set()

    print("\n[INFO] Esecuzione cleanup finale...")
    if "executor" in globals() and executor is not None:
        try:
            executor.shutdown(wait=True)
        except Exception as e:
            print(f"[WARN] Errore durante lo shutdown del thread pool: {e}")


    if all_clip_logs:
        clips_df = pd.DataFrame(all_clip_logs)
        master_log_path = os.path.join(OUTPUT_BASE_DIR, "master_clip_log.csv")
        clips_df.to_csv(master_log_path, index=False)
        print(f"✅ Master clip log with {len(clips_df)} entries saved to: {master_log_path}")

    if all_pipeline_logs and not args.headless:
        log_df = pd.DataFrame(all_pipeline_logs)
        log_df_path = "pipeline_performance_log.csv"
        log_df.to_csv(log_df_path, index=False)
        print(f"✅ Aggregated performance log for {len(log_df)} frames saved to: {log_df_path}")

        plt.figure(figsize=(14, 7))
        plt.plot(log_df.index, log_df["total_pipeline_fps"], label='Total Pipeline FPS', color='b', alpha=0.7)
        plt.title("Overall Pipeline Performance (FPS) Across All Videos")
        plt.xlabel("Frame Number (Overall)")
        plt.ylabel("Frames Per Second (FPS)")
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig("total_pipeline_fps.png")
        plt.show()

        plt.figure(figsize=(14, 9))
        time_cols = ["read_frame_time", "yunet_inference_time", "bytetrack_update_time",
                     "face_preprocessing_time", "au_extraction_time", 
                     "mediapipe_parallel_wall_time", "clip_handling_time", "drawing_time"]
        for col in time_cols:
            if col in log_df.columns:
                plt.plot(log_df.index, log_df[col], label=col.replace("_", " ").title())

        plt.title("Execution Time per Pipeline Component (Seconds)")
        plt.xlabel("Frame Number (Overall)")
        plt.ylabel("Time (seconds)")
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.savefig("time_per_component.png")
        plt.show()

    if not args.headless:
        cv2.destroyAllWindows()
    print("[INFO] Cleanup completato.")

atexit.register(cleanup)

if __name__ == "__main__" :

    executor = None
    all_pipeline_logs = []
    all_clip_logs = []


    # --- Input Source Identification ---
    video_paths = []

    # Caso webcam (solo se non headless)
    if args.input == '0':
        if args.headless:
            print("[ERROR] In modalità headless non è possibile usare la webcam ('0' come input).")
            sys.exit(1)
        else:
            video_paths = ['0']

    # Caso singolo file
    elif os.path.isfile(args.input):
        video_paths = [args.input]

    # Caso directory: cerca ricorsivamente video
    elif os.path.isdir(args.input):
        for root, dirs, files in os.walk(args.input):
            for file in files:
                if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    full_path = os.path.join(root, file)
                    video_paths.append(full_path)
        if not video_paths:
            print(f"[ERROR] Nessun video valido trovato ricorsivamente in '{args.input}'.")
            sys.exit(1)

    # Caso input non valido
    else:
        print(f"[ERROR] Il path fornito non è valido: '{args.input}'")
        sys.exit(1)


    threading.Thread(target=esc_listener, daemon=True).start()

    if not args.headless:
    # --- Preparazione interfaccia di caricamento ---
        cv2.namedWindow("Tracking & Facial Analysis", cv2.WINDOW_NORMAL)
        loading_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(loading_img, "Loading models... Please wait.", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("Tracking & Facial Analysis", loading_img)
        cv2.waitKey(1)

    if _libreface_available:
        try:
            libreface_init_au_model()
            print("LibreFace AU model loaded.")
            #print(f"[DEBUG] args.headless = {args.headless}")
            #print(f"[DEBUG] args.vis = {args.vis}")
            #print(f"[DEBUG] args.backend_target = {args.backend_target}")

        except Exception as e:
            print(f"ERROR: Failed to initialize LibreFace AU model: {e}"); sys.exit(1)

    backend_id, target_id = backend_target_pairs[args.backend_target]
    face_detector = YuNet(modelPath=args.model, inputSize=[640, 480], confThreshold=0.9, nmsThreshold=0.3, topK=500, backendId=backend_id, targetId=target_id)
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    executor = ThreadPoolExecutor(max_workers=args.num_workers_per_frame)

    progress_bar = None


    try:
    # --- Main Loop over Videos ---
        for video_path in video_paths:
            if esc_pressed.is_set():
                print("ESC detected before starting next video. Exiting...")
                break

            source_name = "webcam" if video_path == '0' else os.path.basename(video_path)

            # Reset del contatore ID tracking se disponibile
            if hasattr(STrack, "_count"):
                STrack._count = 0

            cap = cv2.VideoCapture(int(video_path) if video_path == '0' else video_path)
            if not cap.isOpened():
                print(f"Error: Could not open video '{video_path}'. Skipping."); continue

            print(f"\n--- Processing: {source_name} ---")
            w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_writer = None
            if args.output:
                out_path = args.output
                video_writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 15.0, (w, h))
                print(f"[INFO] Output video will be saved to: {out_path}")
            else:
                print(f"[INFO] Output video will NOT be saved.")


            if args.yunet_res > 0:
                aspect_ratio = w / h
                if w > h: new_w, new_h = int(args.yunet_res * aspect_ratio), args.yunet_res
                else: new_w, new_h = args.yunet_res, int(args.yunet_res / aspect_ratio)
                yunet_input_size = [new_w, new_h]
            else:
                yunet_input_size = [w, h]
            face_detector.setInputSize(yunet_input_size)
            
            # --- Per-Video State Initialization ---
            tracker = BYTETracker(Namespace(track_thresh=0.5, match_thresh=0.8, track_buffer=30, frame_rate=30,  mot20=False), frame_rate=30)
            frame_id, active_futures = 0, []
            clip_buffer, au_buffer, land_buffer = {}, {}, {}
            track_clip_counters = {}
            pipeline_logs_for_this_video, video_clip_logs, clips_in_ram = [], [], []

            # Tentativo di calcolo del numero totale di frame (può fallire in streaming)
            try:
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total_frames > 0:
                    progress_bar = tqdm(total=total_frames, desc=f"Processing {source_name}", unit="frame")
                else:
                    progress_bar = None
            except:
                progress_bar = None


            try:
                while cap.isOpened():
                    
                    start_total_frame_time = time.time()
                    frame_log = {"frame_id": frame_id, "source_name": source_name, "clip_handling_time": 0.0}

                    start_read_time = time.time()
                    ret, frame = cap.read() 
                    if not ret: break
                    frame_log["read_frame_time"] = time.time() - start_read_time

                    online_targets, img_h, img_w = detect_and_track(frame, face_detector, tracker, yunet_input_size, frame_log)
                    preprocessed_faces, au_map = preprocess_and_extract_features(frame, online_targets, frame_log)
                    
                    start_mediapipe_wall_time = time.time()
                    if preprocessed_faces and au_map:
                        new_tasks = submit_tasks_to_executor(preprocessed_faces, au_map, executor)
                        active_futures.extend(new_tasks)
                    
                    completed_results, active_futures = collect_completed_futures(active_futures)
                    frame_log["mediapipe_parallel_wall_time"] = time.time() - start_mediapipe_wall_time
                    
                    total_mediapipe_thread_time = 0
                    start_clip_time = time.time()
                    for result in completed_results:
                        total_mediapipe_thread_time += result["processing_time"]
                        handle_clip_buffers(result, clip_buffer, au_buffer, land_buffer, frame_id, clips_in_ram,
                                            video_clip_logs, track_clip_counters, source_name)
                    frame_log["clip_handling_time"] = time.time() - start_clip_time
                    frame_log["mediapipe_thread_time_sum"] = total_mediapipe_thread_time

                    total_frame_time = time.time() - start_total_frame_time
                    frame_log["total_processing_time"] = total_frame_time
                    fps = 1.0 / total_frame_time if total_frame_time > 0 else 0
                    frame_log["total_pipeline_fps"] = fps
                    pipeline_logs_for_this_video.append(frame_log)
                    
                    start_draw_time = time.time()
                    draw_visualizations(frame, preprocessed_faces, completed_results, img_w, img_h, frame_log, frame_id) 
                    frame_log["drawing_time"] = time.time() - start_draw_time  
                    if video_writer:
                        video_writer.write(frame)
        
                    frame_id += 1

                    if progress_bar:
                        progress_bar.update(1)


                    if esc_pressed.is_set():
                        raise InterruptedError("ESC pressed from keyboard input.")

            
            except (KeyboardInterrupt, InterruptedError):
                print("\nInterruption detected. Proceeding to cleanup and final report generation.")
            finally:
                if args.output:
                    print(f"Finished processing '{source_name}'. Processed {frame_id} frames. Saved output to '{args.output}'")
                else:
                    print(f"Finished processing '{source_name}'. Processed {frame_id} frames.")

                cap.release()

                if progress_bar:
                    progress_bar.close()

                if video_writer:
                    video_writer.release()
                BaseTrack._count = 0
                if not args.headless:
                    cv2.destroyAllWindows()
                all_pipeline_logs.extend(pipeline_logs_for_this_video)
                all_clip_logs.extend(video_clip_logs)
    finally:
            # --- Final Cleanup and Reporting (after all videos) ---
            cleanup()