from deepface import DeepFace

# Load and compare two images
result = DeepFace.verify("image1.jpg", "image2.jpeg")

# Print result
print("Verification Result:", result)
