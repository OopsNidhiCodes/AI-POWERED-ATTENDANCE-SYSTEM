from deepface import DeepFace

# Load and compare two images
result = DeepFace.verify("image1.jpeg", "image2.jpg")

# Print result
print("Verification Result:", result)
