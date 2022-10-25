import numpy as np
import argparse
import sys
import cv2

IMSHOW_WINNAME = 'Video'
OUT_VIDEO_PATH = 'out.mp4'
FLANN_INDEX_TREE = 0
FLANN_TREES = 5
FLANN_SEARCH_CHECKS = 50
FLANN_MATCH_DISTANCE = 0.7
K_NEAREST_NEIGHBORS = 2

MATCH_THRESHOLD = 10
RANSAC_THRESHOLD = 5
RESHAPING_VALUES = [-1, 1, 2]


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-i', '--image-path', type=str, required=True, help='Path to the image we will work with')
    parser.add_argument('-v', '--video-path', type=str, required=False, help='Path to the video we will work with')
    parser.add_argument('-a', '--flann-index-algorithm', type=int, required=False, help='Flann index parameters algorithm', default=FLANN_INDEX_TREE)
    parser.add_argument('-t', '--flann-index-trees', type=int, required=False, help='Flann index parameters trees', default=FLANN_TREES)
    parser.add_argument('-s', '--flann-search-checks', type=int, required=False, help='Flann search parameters checks', default=FLANN_SEARCH_CHECKS)
    parser.add_argument('-mt', '--matched-points-threshold', type=int, required=False, help='Matched points threshold value', default=MATCH_THRESHOLD)
    parser.add_argument('-rt', '--ransac-threshold', type=int, required=False, help='Ransac threshold value', default=RANSAC_THRESHOLD)
    parser.add_argument('-md', '--matches-distance', type=float, required=False, help='Max distance between matches', default=FLANN_MATCH_DISTANCE)

    return parser.parse_args()

def match_flann(target_description, scene_description):
    matcher = cv2.FlannBasedMatcher(dict(algorithm=FLANN_INDEX_TREE, trees=FLANN_TREES), dict(checks=FLANN_SEARCH_CHECKS))
    matches = matcher.knnMatch(target_description, scene_description, k=K_NEAREST_NEIGHBORS)
    matched_keypoints = []
    [matched_keypoints.append(m) if m.distance < FLANN_MATCH_DISTANCE * n.distance else None for m, n in matches]
    return matched_keypoints

def draw_matched_frame(ref_image, kp1, frame, kp2, matched_keypoints):
    target_keypoints = np.float32([kp1[matched_keypoint.queryIdx].pt for matched_keypoint in matched_keypoints]).reshape(RESHAPING_VALUES)
    scene_keypoints = np.float32([kp2[matched_keypoint.trainIdx].pt for matched_keypoint in matched_keypoints]).reshape(RESHAPING_VALUES)

    _, mask = cv2.findHomography(target_keypoints, scene_keypoints, cv2.RANSAC, RANSAC_THRESHOLD)
    mask_matches = mask.ravel().tolist()

    x1, y1, w, h = cv2.boundingRect(scene_keypoints)
    start, end = (x1, y1), (x1 + w, y1 + h)
    cv2.rectangle(frame, start, end, [0, 0, 255])

    img = cv2.drawMatches(ref_image, kp1, frame, kp2, matched_keypoints, None,
                          matchColor=(0, 255, 0), singlePointColor=None, matchesMask=mask_matches, flags=2)
    return img

def draw_empty_matches(ref_image, frame):
    img = cv2.drawMatches(ref_image, None, frame, None, None, None,
                          matchColor=(0, 255, 0), singlePointColor=None, matchesMask=None, flags=2)
    return img

def get_keypoints_sift(ref_image, frame):
    ref_image_bw = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
    frame_bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()
    kp1, dsc1 = sift.detectAndCompute(ref_image_bw, None)
    kp2, dsc2 = sift.detectAndCompute(frame_bw, None)
    matched_keypoints = match_flann(dsc1, dsc2)

    if len(matched_keypoints) >= MATCH_THRESHOLD:
        return draw_matched_frame(ref_image, kp1, frame, kp2, matched_keypoints)
    else:
        return draw_empty_matches(ref_image, frame)

def track_features(reference_image, video_capture):
    frame_width, frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ref_width, ref_height = int(frame_width*0.25), int(reference_image.shape[1])
    out_width, out_height = int(ref_width + frame_width), int(frame_height) if frame_height > ref_height else ref_height
    out = cv2.VideoWriter(OUT_VIDEO_PATH, cv2.VideoWriter_fourcc(*'mp4v'),
                          20.0, (out_width, out_height))
    paused = False

    while True:
        if not paused:
            received, frame = video_capture.read()
            frame = cv2.resize(frame, (frame_width, frame_height), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
            reference_image = cv2.resize(reference_image, (ref_width, ref_height), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)

            if not received: break
            processed_frame = get_keypoints_sift(reference_image, frame)
            out.write(processed_frame)
            cv2.imshow(IMSHOW_WINNAME, processed_frame)
        pressed_key = cv2.waitKey(1)
        if pressed_key == ord('q'):
            break
        elif pressed_key == ord('p'):
            paused = not paused

    video_capture.release()
    out.release()
    cv2.destroyAllWindows()

def main():
    args = parse_arguments()
    global FLANN_INDEX_TREE, FLANN_TREES, FLANN_SEARCH_CHECKS, MATCH_THRESHOLD, RANSAC_THRESHOLD
    FLANN_INDEX_TREE = args.flann_index_algorithm
    FLANN_TREES = args.flann_index_trees
    FLANN_SEARCH_CHECKS = args.flann_search_checks
    MATCH_THRESHOLD = args.matched_points_threshold
    RANSAC_THRESHOLD = args.ransac_threshold

    try:
        reference_image = cv2.imread(args.image_path)
        if reference_image is None: raise IOError(f"File {reference_image} doesn't exist or is not a valid image format");

        video_capture = cv2.VideoCapture(args.video_path)
        if not video_capture.isOpened():
            video_capture = cv2.VideoCapture(0)
        track_features(reference_image, video_capture)
        cv2.waitKey()
    except IOError as err:
        sys.exit(err.errno)

if __name__ == '__main__':
    main()