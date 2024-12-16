import cv2
import mediapipe as mp
from djitellopy import Tello
import time

# Inicializa el dron Tello
tello = Tello()
tello.connect()

# Verifica el nivel de batería
battery = tello.get_battery()
if battery < 10:
    print(f"Advertencia: Batería baja ({battery}%)")
    exit()
print(f"Nivel de batería: {battery}%")

# Activa el flujo de video del dron
tello.streamon()
frame_read = tello.get_frame_read()

# Inicializa Mediapipe para detección de manos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Captura desde la cámara del sistema
cap = cv2.VideoCapture(0)

# Funciones auxiliares
def gestos(landmarks):
    punta_pulgar = landmarks[4]
    punta_indice = landmarks[8]
    punta_medio = landmarks[12]
    punta_anular = landmarks[16]
    punta_menique = landmarks[20]
    pulgar_mid = landmarks[3]
    indice_mid = landmarks[6]
    medio_mid = landmarks[10]
    anular_mid = landmarks[14]
    menique_mid = landmarks[18]

    pulgar_arriba = punta_pulgar.x > pulgar_mid.x
    indice_arriba = punta_indice.y < indice_mid.y
    medio_arriba = punta_medio.y < medio_mid.y
    anular_arriba = punta_anular.y < anular_mid.y
    menique_arriba = punta_menique.y < menique_mid.y

    if indice_arriba and not medio_arriba and not anular_arriba and not menique_arriba:
        return "arriba"
    elif indice_arriba and medio_arriba and not anular_arriba and not menique_arriba:
        return "abajo"
    elif pulgar_arriba and indice_arriba and menique_arriba and not medio_arriba and not anular_arriba:
        return "derecha"
    elif indice_arriba and medio_arriba and anular_arriba and not menique_arriba:
        return "izquierda"
    elif not indice_arriba and not medio_arriba and not anular_arriba and not menique_arriba:
        return "para"
    else:
        return "otro"

def verificar_estado(tello):
    try:
        altura = tello.get_height()
        print(f"Altura actual: {altura} cm")
        if altura == 0:
            print("El dron no está en el aire. Intentando despegar nuevamente...")
            tello.takeoff()
    except Exception as e:
        print(f"Error al verificar el estado del dron: {e}")

# Lógica principal
try:
    tello.takeoff()
    time.sleep(2)

    while True:
        # Captura de la cámara del sistema
        ret, frame = cap.read()
        if not ret:
            print("Error al capturar la imagen de la cámara.")
            break

        # Captura de la cámara del dron
        drone_frame = frame_read.frame

        # Procesamiento de la imagen de la cámara del sistema
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultado = hands.process(image_rgb)

        if resultado.multi_hand_landmarks:
            for hand_landmarks in resultado.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                gesto = gestos(hand_landmarks.landmark)

                if gesto == "arriba":
                    verificar_estado(tello)
                    tello.send_control_command("up 30")
                elif gesto == "abajo":
                    verificar_estado(tello)
                    tello.send_control_command("down 30")
                elif gesto == "izquierda":
                    verificar_estado(tello)
                    tello.send_control_command("left 30")
                elif gesto == "derecha":
                    verificar_estado(tello)
                    tello.send_control_command("right 30")
                elif gesto == "para":
                    tello.send_rc_control(0, 0, 0, 0)

        # Muestra las ventanas
        cv2.imshow("Control de dron", frame)
        cv2.imshow("Cámara del dron", drone_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Finalizando control.")

finally:
    try:
        tello.land()
    except Exception as e:
        print(f"Error al aterrizar: {e}")
        tello.emergency()
    cap.release()
    cv2.destroyAllWindows()
    tello.streamoff()
