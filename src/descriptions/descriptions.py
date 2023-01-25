import cv2 as cv
import numpy as np


class DescriptionsTypes:
    ORB = "ORB"
    SIFT = "SIFT"
    SURF = "SURF"


class BaseDescriptor():
    def __init__(self):
        self.matcher = cv.BFMatcher()

    def detect_and_compute(self, img):
        keypoints, descriptors = self.descriptor.detectAndCompute(img, None)
        return keypoints, descriptors

    def match(self, descriptors, template_descriptors):
        matches = self.matcher.match(descriptors, template_descriptors)
        return matches

    def get_matches_img(self,
                        img,
                        keypoints,
                        template_img,
                        template_keypoints,
                        matches,
                        max_keypoints_matches=20
                        ):
        img = cv.drawMatches(img, keypoints,
                             template_img, template_keypoints,
                             matches[:min(max_keypoints_matches, len(matches))], None)
        return img

    def draw_keypoints(self, img, keypoints):
        img = cv.drawKeypoints(img, keypoints, None)
        return img

    def get_embedding(self, descriptions, embedding_size=20, type="first_k"):
        if type == "first_k":
            if len(descriptions) < embedding_size:
                # Дополнение вектора нулями до нужного размера
                descriptions_size = len(descriptions)
                array_zeros = [[0] * len(descriptions[0]) for _ in range(embedding_size - descriptions_size)]
                descriptions = np.concatenate((descriptions, array_zeros), axis=0)

            vector_data = descriptions[:embedding_size, :].reshape(embedding_size * len(descriptions[0]))
        elif type == "bow":
            pass
        else:
            raise NotImplementedError()

        return vector_data


class ORB(BaseDescriptor):
    def __init__(self):
        super(ORB, self).__init__()
        self.descriptor = cv.ORB_create()

    def get_name(self):
        return "ORB"


class SIFT(BaseDescriptor):
    def __init__(self):
        super(SIFT, self).__init__()
        self.descriptor = cv.SIFT_create()

    def get_name(self):
        return "SIFT"


class SURF(BaseDescriptor):
    def __init__(self):
        super(SURF, self).__init__()
        # self.descriptor = cv.SURF_create()
        self.descriptor = cv.xfeatures2d.SURF_create()

    def get_name(self):
        return "SURF"
