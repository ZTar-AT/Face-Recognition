import cv2

import face_recognition

# Load the input image and convert it from BGR to RGB
image = cv2.imread('1.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Load a sample image of the person you want to recognize
sample_image = face_recognition.load_image_file('2.jpg')
sample_image_encoding = face_recognition.face_encodings(sample_image)[0]

# Find all the faces in the input image using the default HOG-based model
face_locations = face_recognition.face_locations(image)
face_encodings = face_recognition.face_encodings(image, face_locations)

# Loop through the detected faces
for face_encoding, face_location in zip(face_encodings, face_locations):
    # See if the face is a match for the known sample image
    match = face_recognition.compare_faces([sample_image_encoding], face_encoding)
    name = "Unknown"

    if match[0]:
        name = "T'Challa"

    # Draw a box around the face
    top, right, bottom, left = face_location
    cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)

    # Draw a label with the person's name below the face
    cv2.rectangle(image, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(image, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

# Show the output image
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (960, 540))                # Resize image
cv2.imshow('Face Recognition', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
