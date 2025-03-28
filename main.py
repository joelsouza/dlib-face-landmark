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


def create_collage(images, original_image):
    # Usa as dimensões da imagem original
    height, width = original_image.shape[:2]

    # Cria a imagem da colagem com o tamanho da imagem original
    collage = np.zeros((height, width, 3), dtype=np.uint8)

    n_images = len(images)
    if n_images == 0:
        return collage

    # Cálculo de linhas e colunas para o grid
    n_cols = int(math.ceil(math.sqrt(n_images)))
    n_rows = int(math.ceil(n_images / n_cols))

    # Calcula o tamanho de cada célula na grade
    cell_width = width // n_cols
    cell_height = height // n_rows

    # Posiciona cada imagem na colagem
    idx = 0
    for i in range(n_rows):
        for j in range(n_cols):
            if idx < n_images:
                img = images[idx]
                h, w = img.shape[:2]

                # Redimensiona a imagem para caber na célula, mantendo a proporção
                scale = min(cell_width / w, cell_height / h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                resized_img = cv2.resize(img, (new_w, new_h))

                # Posição da imagem na colagem
                y_offset = i * cell_height
                x_offset = j * cell_width

                # Centraliza a imagem em sua célula
                y_center = y_offset + (cell_height - new_h) // 2
                x_center = x_offset + (cell_width - new_w) // 2

                # Coloca a imagem na colagem
                collage[y_center:y_center+new_h, x_center:x_center+new_w] = resized_img

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
        collage = create_collage(masked_faces, image)  # Passa a imagem original como referência
        cv2.imwrite('faces_collage.jpg', cv2.cvtColor(collage, cv2.COLOR_RGB2BGR))
        print(f"Colagem criada com {len(masked_faces)} faces.")
    else:
        print("Nenhuma face detectada na imagem.")

if __name__ == '__main__':
    args = sys.argv[1:]
    main(args[0])
