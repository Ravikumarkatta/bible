import os
import json

# ...existing code...

# Load the configuration
with open('config/data_config.json', 'r') as f:
    data_config = json.load(f)

embeddings_path = data_config['embeddings_path']

# Create the directory if it doesn't exist
if not os.path.exists(embeddings_path):
    os.makedirs(embeddings_path)
    print(f"Created directory: {embeddings_path}")
else:
    print(f"Directory already exists: {embeddings_path}")

# ...rest of your code...