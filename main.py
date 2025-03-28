import dlib
import cv2
import numpy as np
import sys
import math

def get_landmarks(shape):
    landmarks = []
    for i in range(shape.num_parts):
        landmarks.append((shape.part(i).x, shape.part(i).y))
    return landmarks


def get_routes(shape):
    routes = [i for i in range(
        16, -1, -1)] + [i for i in range(17, 19)] + [i for i in range(24, 26)] + [16]
    return routes


def masked_image(image, routes_coordinates):
    # desenha a mascara
    mask = np.zeros((image.shape[0], image.shape[1]))

    # desenha o contorno do rosto
    mask = cv2.fillPoly(mask, [np.array(routes_coordinates)], 1)
    mask = mask.astype(bool)

    # aplica a mascara na imagem original
    out = np.zeros_like(image)
    out[mask] = image[mask]
    return out


def create_collage(images):
    # Determina o número de linhas e colunas para a colagem
    n_images = len(images)
    if n_images == 0:
        return None

    # Cálculo de linhas e colunas para o grid
    n_cols = int(math.ceil(math.sqrt(n_images)))
    n_rows = int(math.ceil(n_images / n_cols))

    # Encontra a maior altura e largura entre as imagens
    max_height = max(img.shape[0] for img in images)
    max_width = max(img.shape[1] for img in images)

    # Cria a imagem da colagem
    collage = np.zeros((max_height * n_rows, max_width * n_cols, 3), dtype=np.uint8)

    # Posiciona cada imagem na colagem
    idx = 0
    for i in range(n_rows):
        for j in range(n_cols):
            if idx < n_images:
                img = images[idx]
                h, w = img.shape[:2]

                # Posição da imagem na colagem
                y_offset = i * max_height
                x_offset = j * max_width

                # Centraliza a imagem em seu espaço na grade
                y_center = y_offset + (max_height - h) // 2
                x_center = x_offset + (max_width - w) // 2

                # Coloca a imagem na colagem
                collage[y_center:y_center+h, x_center:x_center+w] = img

                idx += 1

    return collage


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

def main(image_path):
    image = dlib.load_rgb_image(image_path)
    faces = detector(image, 1)

    # Lista para armazenar todas as faces mascaradas
    masked_faces = []

    for k, face in enumerate(faces):
        shape = predictor(image, face)
        landmarks = get_landmarks(shape)
        routes = get_routes(shape)

        routes_coordinates = [(landmarks[i][0], landmarks[i][1]) for i in routes]
        copy_image = image.copy()
        generated_image = masked_image(copy_image, routes_coordinates)

        # Adiciona a face mascarada à lista, em vez de salvar individualmente
        masked_faces.append(generated_image)

    # Cria uma colagem com todas as faces
    if masked_faces:
        collage = create_collage(masked_faces)
        cv2.imwrite('faces_collage.jpg', cv2.cvtColor(collage, cv2.COLOR_RGB2BGR))
        print(f"Colagem criada com {len(masked_faces)} faces.")
    else:
        print("Nenhuma face detectada na imagem.")

if __name__ == '__main__':
    args = sys.argv[1:]
    main(args[0])
