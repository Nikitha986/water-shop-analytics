from deepface import DeepFace
import uuid

known_faces = {}

def identify_customer(face_img):
    for cid, embedding in known_faces.items():
        try:
            result = DeepFace.verify(face_img, embedding, enforce_detection=False)
            if result["verified"]:
                return cid
        except:
            pass

    new_id = str(uuid.uuid4())[:8]
    known_faces[new_id] = face_img
    return new_id
