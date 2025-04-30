import numpy as np

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    v1, v2 = a - b, c - b
    cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def extract_angles(keypoints, hand_keypoints):
    def get_coords(index):
        if keypoints is None or len(keypoints) < (index + 1):
            return None
        if keypoints[index][2] > 0:
            return (keypoints[index][0], keypoints[index][1])
        return None

    def get_hand_coords(index):
        if hand_keypoints is None or len(hand_keypoints) < (index + 1):
            return None
        if hand_keypoints[index][2] > 0:
            return (hand_keypoints[index][0], hand_keypoints[index][1])
        return None

    hip = get_coords(9)
    knee = get_coords(10)
    ankle = get_coords(11)
    shoulder = get_coords(2)
    elbow = get_coords(3)
    wrist = get_coords(4)
    neck = get_coords(1)
    head = get_coords(17)
    back = get_coords(8)

    thumb = get_hand_coords(4)
    pinky = get_hand_coords(20)

    angles = {
        "knee_angle": calculate_angle(hip, knee, ankle) if hip and knee and ankle else None,
        "elbow_angle": calculate_angle(shoulder, elbow, wrist) if shoulder and elbow and wrist else None,
        "neck_angle": calculate_angle(back, neck, head) if back and neck and head else None,
        "thigh_back_angle": calculate_angle(knee, hip, neck) if knee and hip and neck else None,
        "wrist_angle": calculate_angle(thumb, wrist, pinky) if wrist and thumb and pinky else None,
        "shoulder_angle": calculate_angle(back, shoulder, elbow) if back and shoulder and elbow else None,
    }
    return angles
