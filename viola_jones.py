import sys
import cv2
import numpy as np
import pickle as pkl
import matplotlib as mpl
mpl.use('TkAgg')
mpl.rcParams['figure.facecolor'] = '1.0'
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


class haar_like_feature:
    def __init__(self, lower_left_x, lower_left_y, width, height, orientation):
        self.ul_x = lower_left_x
        self.ul_y = lower_left_y
        self.w = width
        self.h = height
        self.o = orientation

    def intensity(self, integral_image, ul_x, ul_y, w, h):
        # print "lr = {}, ul = {}, ur = {}, ll = {}".format(
        #     integral_image[ul_y + h, ul_x + w],
        #     integral_image[ul_y,     ul_x    ],
        #     integral_image[ul_y,     ul_x + w],
        #     integral_image[ul_y + h, ul_x    ]
        # )
        return integral_image[ul_y + h, ul_x + w] + \
               integral_image[ul_y,     ul_x    ] - \
               integral_image[ul_y,     ul_x + w] - \
               integral_image[ul_y + h, ul_x    ]

    def value(self, integral_image):
        if self.o == "vertical":
            left_intensity = self.intensity(
                integral_image,
                self.ul_x,
                self.ul_y,
                self.w/2,
                self.h
                )
            right_intensity = self.intensity(
                integral_image,
                self.ul_x + self.w/2,
                self.ul_y,
                self.w/2,
                self.h
                )
            # print "ul = ({},{}), width = {}, height = {}".format(self.ul_x, self.ul_y, self.w, self.h)
            # print "left_intensity = {}".format(left_intensity)
            # print "right_intensity = {}".format(right_intensity)
            return left_intensity - right_intensity

        else:
            top_intensity = self.intensity(
                integral_image,
                self.ul_x,
                self.ul_y,
                self.w,
                self.h/2
                )
            bottom_intensity = self.intensity(
                integral_image,
                self.ul_x,
                self.ul_y + self.h/2,
                self.w,
                self.h/2
                )
            # print "ul = ({},{}), width = {}, height = {}".format(self.ul_x, self.ul_y, self.w, self.h)
            # print "top_intensity = {}".format(top_intensity)
            # print "bottom_intensity = {}".format(bottom_intensity)
            return top_intensity - bottom_intensity


def classifier_value(image, feature, polarity, threshold):
    return np.sign(polarity * (feature.value(image) - threshold))


def strong_classifier_value(image, features, feature_indices, weights, polarities, thresholds, meta_threshold):
    return np.sign(
        strong_classifier_sum(image, features, feature_indices, weights, polarities, thresholds, meta_threshold)
    )
def strong_classifier_sum(image, features, feature_indices, weights, polarities, thresholds, meta_threshold):
    # number of weak classifiers
    T = len(feature_indices)

    s = sum(
        [weights[t] * classifier_value(
            image,
            features[int(feature_indices[t])],
            polarities[int(feature_indices[t])],
            thresholds[int(feature_indices[t])]
        ) for t in range(T)]
    ) - meta_threshold

    # print s

    return s


def generate_integral_image(image):
    ii = np.empty(image.shape)

    for i in range(image.shape[0]):
        s = 0
        for j in range(image.shape[1]):
            s += image[i,j]

            if i == 0:
                ii[i,j] = s
            else:
                ii[i,j] = ii[i-1,j] + s

    return ii


def polarity_threshold(images, labels, feature, D_t):
    # number of training images
    m = images.shape[0]
    # permutation ordering examples by feature values from low to high
    sigma = sorted(range(m), key=lambda j: feature.value(images[j]))

    errors = np.empty(m)
    polarities = np.empty(m)

    # positive examples on left side of j
    L_plus  = 0
    # megative examples on left side of j
    L_minus = 0
    # positive examples on right side of j
    R_plus  = sum([D_t[sigma[k]]*(labels[sigma[k]] == 1)  for k in range(0, m)])
    # negative examples on right side of j
    R_minus = sum([D_t[sigma[k]]*(labels[sigma[k]] == -1) for k in range(0, m)])

    for j in range(m):
        if labels[sigma[j]] == 1:
            L_plus += D_t[sigma[j]]
            R_plus -= D_t[sigma[j]]
        else:
            L_minus += D_t[sigma[j]]
            R_minus -= D_t[sigma[j]]
        # If more positive weight is on the right of sigma[j], polarity is positive
        polarities[sigma[j]] = 1 if L_plus - L_minus < R_plus - R_minus else -1
        # weight of misclassified examples if threshold is set to f(sigma[j])
        errors[sigma[j]] = L_plus + R_minus if polarities[sigma[j]] == 1 else L_minus + R_plus

    j_optimal = min(range(m), key=lambda j: errors[j])

    # set polarity to that used to minimize error for j_optimal
    polarity = polarities[j_optimal]
    # set threshold to value of feature
    threshold = feature.value(images[j_optimal])

    # print errors
    # print "j_optimal = {}, polarity = {}, threshold = {}".format(j_optimal, polarity, threshold)
    # raw_input()

    return polarity, threshold


def adaboost(images, labels, features, T):
    # number of training images
    m = images.shape[0]
    # number of features
    N = len(features)
    # distribution over examples
    D_t = np.full(m, 1.0/m)
    # weights for base features within overall classifier
    weights = []
    # features to use
    feature_indices = []
    # polarities and thresholds for each feature
    polarities = np.empty(N)
    thresholds = np.empty(N)

    for t in range(T):
        print "    Training weak classifier t = {}".format(t)
        # find best polarity and threshold for each feature
        print "\tDetermining optimal polarity and threshold for each feature"
        for i in range(N):
            polarities[i], thresholds[i] = polarity_threshold(images, labels, features[i], D_t)

        print "\tDetermining error for each feature"
        # epsilon_t for each classifier
        errors = np.array([
            sum(
                [D_t[i] if classifier_value(images[i], features[j], polarities[j], thresholds[j]) != labels[i] else 0 for i in range(m)]
            ) for j in range(N)
        ])
        # print errors
        # index of best classifier and lowest error
        c = min(range(N), key=lambda j: errors[j])
        # if this feature has already been used, then we've separated the dataset as much as possible
        # so keep the set of weak classifiers that we currently have without adding more
        if c in feature_indices[:t]:
            print "\tPREVIOUSLY SELECTED FEATURE ENCOUNTERED"
            break
        # base classifier with smallest error
        feature_indices.append(c)
        print "\tfeature index = {}".format(c)
        # smallest error
        epsilon_t = errors[c]
        # print "\tepsilon_t = {}".format(epsilon_t)
        # weight to use for this classifier
        weights.append(0.5*np.log((1-epsilon_t)/epsilon_t))
        # normalization factor
        Z_t = 2.0*(epsilon_t*(1-epsilon_t))**0.5
        # print "\tZ_t = {}".format(Z_t)
        # distribution over examples
        D_t = np.array([
            D_t[i] * np.exp(-1.0 * weights[t] * labels[i] * classifier_value(images[i], features[c], polarities[j], thresholds[j])) / Z_t for i in range(m)
        ])

    return feature_indices, weights, polarities, thresholds


def detect_faces(image, h):
    height, width = image.shape

    detected = []

    stride = 4

    for ul_x in range(0, width - 64, stride):
        for ul_y in range(0, height - 64, stride):
            if h(image[ul_y:ul_y+64, ul_x:ul_x+64]) == 1:
                detected.append(np.array([ul_x, ul_y]))

    return detected


def write_detections(image, detected):
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    for ul in detected:
        cv2.rectangle(image, tuple(ul), tuple(ul+np.array([64,64])), (255,0,0), 3)

    return image


def get_meta_threshold(images, labels, features, feature_indices, weights, polarities, thresholds):
    # number of examples
    m = images.shape[0]
    # number of weak classifiers
    T = len(feature_indices)

    # for t in range(T):
    #     print "t = {}".format(t)
    #     print "  weight = {}".format(weights[t])
    #     print "  feature index = {}".format(int(feature_indices[t]))
    #     print "  polarity = {}, threshold = {}".format(polarities[int(feature_indices[t])], thresholds[int(feature_indices[t])])
    #     raw_input()

    return min([strong_classifier_sum(
        images[i],
        features,
        feature_indices,
        weights,
        polarities,
        thresholds,
        0) if labels[i] == 1 else np.inf for i in range(m)])


def filter_negatives(images, labels, h):
    # number of examples
    m = images.shape[0]

    filtered_images = []
    filtered_labels = []

    for i in range(m):
        if h(images[i]) == 1:
            filtered_images.append(images[i])
            filtered_labels.append(labels[i])

    return np.array(filtered_images), np.array(filtered_labels)


def false_positive_rate(images, labels, h):
    # number of examples
    m = images.shape[0]

    return sum([h(images[i]) == 1 and labels[i] == -1 for i in range(m)]) / float(m)


def main(run="test", n=4000):
    if run == "test":
        print "Generating Haar-like features..."
        features = []
        stride = 16
        # for width in range(4, 65, stride):
        #     for height in range(4, 65, stride):
        for width in [4, 6, 8, 12, 16, 24, 32]: # [4, 8, 16, 32]: #
            for height in [4, 6, 8, 12, 16, 24, 32]: # [4, 8, 16, 32]: #
                for ul_x in range(0, 64 - width, stride):
                    for ul_y in range(0, 64 - height, stride):
                        for orientation in ["vertical", "horizontal"]:
                            features.append(
                                haar_like_feature(
                                    ul_x, ul_y,
                                    width, height,
                                    orientation
                                )
                            )
        print "  Using {} Haar-like features".format(len(features))

        print "Deserializing training integral images..."
        with open("faces_ii.pkl", "rb") as face_pkl, open("background_ii.pkl", "rb") as bkgd_pkl:
            images = np.concatenate((pkl.load(face_pkl), pkl.load(bkgd_pkl)))
            labels = np.concatenate((np.full(2000, 1), np.full(2000, -1)))

        # take fewer training examples for testing
        all_images = np.concatenate((images[:n/2], images[2000:2000+n/2]))
        all_labels = np.concatenate((labels[:n/2], labels[2000:2000+n/2]))
        images = all_images
        labels = all_labels

        # open test image and calculate integral image
        test_image = cv2.imread("class.jpg", 0)
        test_image_ii = generate_integral_image(test_image)

        # number of features to select in each strong classifier runs
        Ts = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # strong classifiers from each round of the cascade
        params = []

        for i in range(len(Ts)):
            print "Cascade round {}".format(i+1)
            print "  Training Adaboost classifier..."
            feature_indices, weights, polarities, thresholds = adaboost(images, labels, features, Ts[i])

            print "  Determining meta-threshold so there are no false negatives..."
            meta_threshold = get_meta_threshold(images, labels, features, feature_indices, weights, polarities, thresholds)
            print "\tmeta_threshold = {}".format(meta_threshold)

            # make callable strong classifier
            h = lambda image: strong_classifier_value(image, features, feature_indices, weights, polarities, thresholds, meta_threshold)

            print "    Total false positive rate = {}".format(false_positive_rate(all_images, all_labels, h))

            # add h to collection of strong classifiers
            params.append([feature_indices, weights, polarities, thresholds, meta_threshold])

            if i+1 != len(Ts):
                print "  Filtering out classified non-faces..."
                images, labels = filter_negatives(images, labels, h)
                print "\tnumber of remaining training images = {}".format(images.shape[0])

        # combined classifier
        h_all = lambda image: 1 if all([strong_classifier_value(image, features, *params[i]) == 1 for i in range(len(Ts))]) else -1
        print "Combined false positive rate = {}".format(false_positive_rate(all_images, all_labels, h_all))

        print "  Running classifier on test image..."
        # get coordinates of upper left for each classified face
        detected = detect_faces(test_image_ii, h_all)

        print "    {} faces detected".format(len(detected))
        # display faces
        test_image = write_detections(test_image, detected)

        cv2.imwrite("class_faces.jpg", cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR))

        # imgplot = plt.imshow(test_image)
        # plt.show()

    # compute integral image for each training example
    elif run == "ii":
        faces = []
        background = []

        for i in range(2000):
            face_name = "faces/face{}.jpg".format(i)
            background_name = "background/{}.jpg".format(i)

            faces.append(
                generate_integral_image(
                    cv2.imread(face_name, 0)
                )
            )

            background.append(
                generate_integral_image(
                    cv2.imread(background_name, 0)
                )
            )

        with open("faces_ii.pkl", "wb") as f:
            pkl.dump(faces, f)

        with open("background_ii.pkl", "wb") as f:
            pkl.dump(background, f)

    # test Haar-like features
    elif run == "haar":
        feature_vert = haar_like_feature(
            0, 0,
            64, 64,
            "vertical"
        )
        feature_horiz = haar_like_feature(
            0, 0,
            64, 64,
            "horizontal"
        )

        image_vert = cv2.imread("haar_test_vert.jpg", 0)
        image_horiz = cv2.imread("haar_test_horiz.jpg", 0)

        image_vert_ii = generate_integral_image(image_vert)
        image_horiz_ii = generate_integral_image(image_horiz)

        print "Vertical feature on vertical image = {}".format(feature_vert.value(image_vert_ii))
        print "Vertical feature on horizontal image = {}".format(feature_vert.value(image_horiz_ii))
        print "Horizontal feature on vertical image = {}".format(feature_horiz.value(image_vert_ii))
        print "Horizontal feature on horizontal image = {}".format(feature_horiz.value(image_horiz_ii))


if __name__ == "__main__":
    if len(sys.argv) > 2:
        main(run=sys.argv[1], n=int(sys.argv[2]))
    elif len(sys.argv) > 1:
        main(run=sys.argv[1])
    else:
        main()
