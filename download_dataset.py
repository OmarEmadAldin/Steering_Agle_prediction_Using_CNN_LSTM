import kagglehub

# Download latest version
path = kagglehub.dataset_download("phamvoquoclong/steering-dataset")

print("Path to dataset files:", path)
