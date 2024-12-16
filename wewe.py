import cv2
import mediapipe as mp
from djitellopy import Tello
import time

# Inicializa el dron Tello
tello = Tello()
tello.connect()
print(f"Nivel de batería: {tello.get_battery()}%")

# Inicializa Mediapipe para la detección de manos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Captura desde la cámara del sistema
cap = cv2.VideoCapture(0)


# Control de gestos
def gestos(landmarks):
    # Coordenadas de las puntas de los dedos
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

    # Comprueba si los dedos están levantados
    pulgar_arriba = punta_pulgar.x > pulgar_mid.x  # El pulgar apunta hacia la derecha (mano derecha)
    indice_arriba = punta_indice.y < indice_mid.y
    medio_arriba = punta_medio.y < medio_mid.y
    anular_arriba = punta_anular.y < anular_mid.y
    menique_arriba = punta_menique.y < menique_mid.y

    # Detecta gestos específicos
    if indice_arriba and not medio_arriba and not anular_arriba and not menique_arriba:  # Solo índice levantado
        return "arriba"
    elif indice_arriba and medio_arriba and not anular_arriba and not menique_arriba:  # Índice y medio levantados
        return "abajo"
    elif pulgar_arriba and indice_arriba and menique_arriba and not medio_arriba and not anular_arriba:  # Pulgar, índice y meñique levantados
        return "derecha"
    elif indice_arriba and medio_arriba and anular_arriba and not menique_arriba:  # Índice, medio y anular levantados
        return "izquierda"

    elif indice_arriba and medio_arriba and anular_arriba and menique_arriba and not pulgar_arriba:  # Índice, medio y anular levantados
        return "adelante"

    elif indice_arriba and medio_arriba and anular_arriba and menique_arriba and pulgar_arriba:  # Índice, medio y anular levantados
        return "atras"



    elif not indice_arriba and not medio_arriba and not anular_arriba and not menique_arriba:  # Ningún dedo levantado
        return "para"
    else:
        return "otro"


# Simulación del comportamiento
try:
    tello.takeoff()  # Despega el dron
    time.sleep(2)

    print("Control de gestos activado. Usa gestos para controlar el dron.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Procesa la imagen para detección de manos
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultado = hands.process(image_rgb)

        if resultado.multi_hand_landmarks:
            for hand_landmarks in resultado.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                gesto = gestos(hand_landmarks.landmark)

                # Control del dron basado en gestos
                if gesto == "para":
                    print("Detenido")
                    tello.send_rc_control(0, 0, 0, 0)  # Detener el dron
                elif gesto == "arriba":
                    print("Subiendo")
                    tello.move_up(50)
                elif gesto == "abajo":
                    print("Bajando")
                    tello.move_down(50)
                elif gesto == "derecha":
                    print("Moviéndose a la derecha")
                    tello.move_right(50)
                elif gesto == "izquierda":
                    print("Moviéndose a la izquierda")
                    tello.move_left(50)
                elif gesto == "adelante":
                    print("Moviéndose adelante")
                    tello.move_forward(50)

                elif gesto == "atras":
                    print("Moviéndose atras")
                    tello.move_back(50)
                elif gesto == "otro":
                    print("Otro gesto detectado")

        # Muestra la cámara
        cv2.imshow("Control de dron por gestos", frame)

        # Presiona 'q' para salir
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Simulación terminada.")

finally:
    tello.land()  # Aterriza el dron
    cap.release()
    cv2.destroyAllWindows()
    print("Cámara cerrada y dron aterrizado.")