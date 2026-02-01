import cv2
import mediapipe as mp
import numpy as np
import os
import time

# --- caminhos dos modelos e filtros ---
MODELS_PATH = "models/"
FILTERS_PATH = "filters/"

def load_img(name):
    """carrega imagem com transparência"""
    img = cv2.imread(os.path.join(FILTERS_PATH, name), cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"não achei {name}")
    return img  # sem redimensionamento aqui

# --- carrega os filtros ---
glasses = load_img("glasses.png")
moust = load_img("moustache.png")
mask = load_img("mask.png")
crown = load_img("crown.png")

# --- inicializa mediapipe ---
BaseOptions = mp.tasks.BaseOptions
VisionMode = mp.tasks.vision.RunningMode

# detector de rosto
face_opts = mp.tasks.vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=f'{MODELS_PATH}face_landmarker.task'),
    running_mode=VisionMode.VIDEO
)

# reconhecedor de gestos
gest_opts = mp.tasks.vision.GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=f'{MODELS_PATH}gesture_recognizer.task'),
    running_mode=VisionMode.VIDEO
)

def overlay_transparent(frame, filt, x, y, scale=1):
    """sobrepõe o filtro no frame usando transparência"""
    if filt is None:
        return frame  # nada a fazer

    # redimensiona o filtro
    h, w = filt.shape[:2]
    nw, nh = int(w*scale), int(h*scale)
    if nw <= 0 or nh <= 0:
        return frame

    filt_r = cv2.resize(filt, (nw, nh))

    # define a área do frame onde o filtro vai aparecer
    y1, y2 = max(0, y), min(frame.shape[0], y+nh)
    x1, x2 = max(0, x), min(frame.shape[1], x+nw)

    # define qual parte do filtro cabe na tela
    ov_y1, ov_y2 = max(0, -y), min(nh, frame.shape[0]-y)
    ov_x1, ov_x2 = max(0, -x), min(nw, frame.shape[1]-x)

    if x2-x1 <= 0 or y2-y1 <= 0:
        return frame  # nada visível, pula

    # recorta o frame e o filtro
    roi = frame[y1:y2, x1:x2]
    part = filt_r[ov_y1:ov_y2, ov_x1:ov_x2]

    # aplica transparência se houver
    if part.shape[2] == 4:
        transparency = part[:, :, 3] / 255.0
        transparency = transparency[..., None]  # para multiplicar com RGB
        frame[y1:y2, x1:x2] = roi*(1-transparency) + part[:, :, :3]*transparency
    else:
        frame[y1:y2, x1:x2] = part[:, :, :3]  # sem transparência

    return frame

# --- captura da webcam ---
cap = cv2.VideoCapture(0)

with mp.tasks.vision.FaceLandmarker.create_from_options(face_opts) as face_det, \
     mp.tasks.vision.GestureRecognizer.create_from_options(gest_opts) as gest_det:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # converte o frame pro formato do mediapipe
        ts = int(time.time()*1000)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # detecta rostos e gestos
        faces = face_det.detect_for_video(mp_img, ts)
        gestures = gest_det.recognize_for_video(mp_img, ts)

        # pega o gesto ativo
        active_gesture = "None"
        if gestures.gestures and len(gestures.gestures[0]) > 0:
            active_gesture = gestures.gestures[0][0].category_name

        if faces.face_landmarks:
            f = faces.face_landmarks[0]
            h_fr, w_fr, _ = frame.shape

            # calcula distância entre olhos pra ajustar escala dos filtros
            d_eyes = np.linalg.norm(np.array([f[33].x, f[33].y]) - np.array([f[263].x, f[263].y])) * w_fr

            if active_gesture == "Victory":
                # mascara centralizada no nariz
                w_target = d_eyes * 2.8
                sc = w_target / mask.shape[1]
                x = int(f[1].x*w_fr - (mask.shape[1]*sc/2))
                y = int(f[1].y*h_fr - (mask.shape[0]*sc/2))
                frame = overlay_transparent(frame, mask, x, y, sc)

            elif active_gesture == "Thumb_Up":
                # coroa na cabeça
                w_target = d_eyes * 2.5
                sc = w_target / crown.shape[1]
                x = int(f[10].x*w_fr - (crown.shape[1]*sc/2))
                y = int(f[10].y*h_fr - (crown.shape[0]*sc))
                frame = overlay_transparent(frame, crown, x, y, sc)

            else:
                # aoculos sobre os olhos
                lx, rx = f[33], f[263]
                cx, cy = int((lx.x+rx.x)/2*w_fr), int((lx.y+rx.y)/2*h_fr)
                sc = (d_eyes*2.2)/glasses.shape[1]
                frame = overlay_transparent(frame, glasses, int(cx-(glasses.shape[1]*sc/2)), int(cy-(glasses.shape[0]*sc/2)), sc)

                # bigode entre nariz e boca
                nose, lip = f[1], f[164]
                cx_m, cy_m = int(nose.x*w_fr), int((nose.y+lip.y)/2*h_fr)
                sc_m = (d_eyes*1.2)/moust.shape[1]
                frame = overlay_transparent(frame, moust, int(cx_m-(moust.shape[1]*sc_m/2)), int(cy_m-(moust.shape[0]*sc_m/2)), sc_m)

        # mostra resultado final
        cv2.imshow("Filtro", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
