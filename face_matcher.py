import face_recognition
def get_match_score(distance, threshold=0.6):
    if distance > threshold:
        return 0.0
    score = (1.0 - distance / threshold) * 100
    return round(score, 2)

def compare_faces(file1, file2, tolerance=0.6):
    try:
        # Load images from uploaded files
        image1 = face_recognition.load_image_file(file1)
        image2 = face_recognition.load_image_file(file2)

        # Get face encodings (like vector fingerprints of faces)
        encodings1 = face_recognition.face_encodings(image1)
        encodings2 = face_recognition.face_encodings(image2)

        # If face not detected in either image, return False
        if not encodings1 or not encodings2:
            return False

        # Compare the two face encodings
        distance = face_recognition.face_distance([encodings1[0]],encodings2[0])[0]
        score = get_match_score(distance)
        result = face_recognition.compare_faces([encodings1[0]], encodings2[0], tolerance)
        out = [result[0],score]
  
        return out
    except Exception as e:
        print(f"Error during face comparison: {e}")
        return False
