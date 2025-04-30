import os
from openpose import pyopenpose as op

params = {
    "model_pose": "BODY_25",
    "hand": True,
    "number_people_max": 1,
    "disable_multi_thread": True,
    "model_folder": "/root/openpose/models"
}

opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

def process_image_with_openpose(image):
    datum = op.Datum()
    datum.cvInputData = image
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))

    keypoints = datum.poseKeypoints[0] if datum.poseKeypoints is not None else []
    hand_keypoints = datum.handKeypoints[1][0] if datum.handKeypoints and len(datum.handKeypoints[1]) > 0 else []
    return keypoints, hand_keypoints
