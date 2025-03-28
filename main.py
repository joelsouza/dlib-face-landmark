import dlib
import cv2
import numpy as np
import sys

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


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

def main(image_path):
    image = dlib.load_rgb_image(image_path)
    faces = detector(image, 1)
    for k, face in enumerate(faces):
        shape = predictor(image, face)
        landmarks = get_landmarks(shape)
        routes = get_routes(shape)
        shape = predictor(image, face)
        routes_coordinates = [(landmarks[i][0], landmarks[i][1]) for i in routes]
        copy_image = image.copy()
        generated_image = masked_image(copy_image, routes_coordinates)
        cv2.imwrite(f'masked_face_{k}.jpg', cv2.cvtColor(
            generated_image, cv2.COLOR_RGB2BGR))

if __name__ == '__main__':
    args = sys.argv[1:]
    main(args[0])
