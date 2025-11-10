import torch
import sys

def read_pt_file(file_path):
    try:
        # Load the serialized object from the .pt file
        data = torch.load(file_path, map_location='cpu') # Use 'cpu' if you don't have a GPU or want to load on CPU

        # Check if the loaded data is a dictionary (common for state_dict)
        if isinstance(data, dict):
            print(f"Keys in the .pt file (state_dict): {data.keys()}")
            # You can also print specific values or structure if needed
            # For example, to see the shape of a specific weight:
            # if 'transformer.patch_embed.proj.bias' in data:
            #     print(f"Shape of 'transformer.patch_embed.proj.bias': {data['transformer.patch_embed.proj.bias'].shape}")
        else:
            print(f"The .pt file contains an object of type: {type(data)}")
            print("Content of the .pt file:")
            print(data)

    except Exception as e:
        print(f"Error reading .pt file: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python read_pt_file.py <path_to_pt_file>")
        sys.exit(1)

    file_path = sys.argv[1]
    read_pt_file(file_path)