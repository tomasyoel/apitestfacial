from flask import Flask, request, jsonify
import cv2
import numpy as np
import os

app = Flask(__name__)

@app.route('/compare_faces', methods=['POST'])
def compare_faces():
    # Recibir las imágenes desde la solicitud (DNI y Selfie)
    dni_image_data = request.files.get('dni_image').read()
    selfie_image_data = request.files.get('selfie_image').read()

    # Convertir imágenes a formato que OpenCV puede procesar
    dni_image = cv2.imdecode(np.frombuffer(dni_image_data, np.uint8), cv2.IMREAD_COLOR)
    selfie_image = cv2.imdecode(np.frombuffer(selfie_image_data, np.uint8), cv2.IMREAD_COLOR)

    # Realizar procesamiento y comparación
    is_match = check_similarity(dni_image, selfie_image)
    
    # Responder con el resultado
    return jsonify({'match': is_match})

def check_similarity(image1, image2):
    # Convertir a escala de grises
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Detector de características (SIFT)
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # Comparar características
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Aplicar filtro de Lowe para buenos emparejamientos
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
    return len(good_matches) > 10  # Ajusta el umbral según tus necesidades

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)





# from flask import Flask, request, jsonify
# import cv2
# import numpy as np

# app = Flask(__name__)

# @app.route('/compare_faces', methods=['POST'])
# def compare_faces():
#     dni_image_data = request.files.get('dni_image').read()
#     selfie_image_data = request.files.get('selfie_image').read()

#     dni_image = cv2.imdecode(np.frombuffer(dni_image_data, np.uint8), cv2.IMREAD_COLOR)
#     selfie_image = cv2.imdecode(np.frombuffer(selfie_image_data, np.uint8), cv2.IMREAD_COLOR)

#     is_match = check_similarity(dni_image, selfie_image)
#     return jsonify({'match': is_match})

# def check_similarity(image1, image2):
#     gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
#     gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

#     sift = cv2.SIFT_create()
#     kp1, des1 = sift.detectAndCompute(gray1, None)
#     kp2, des2 = sift.detectAndCompute(gray2, None)

#     bf = cv2.BFMatcher()
#     matches = bf.knnMatch(des1, des2, k=2)
#     good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
#     return len(good_matches) > 10  # Ajusta el umbral según pruebas

# if __name__ == '__main__':
#     app.run(debug=True)
