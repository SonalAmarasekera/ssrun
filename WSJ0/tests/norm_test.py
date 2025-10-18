import torch

# Path to your latent file
latent_file_path = "/content/latents/min/test/mix_clean/447o030v_0.1232_050c0109_-0.1232.pt"  # <--- CHANGE THIS

# Load the dictionary from the file
data_dict = torch.load(latent_file_path)

# 1. See what's inside the dictionary
print(f"Loaded object is a: {type(data_dict)}")
if isinstance(data_dict, dict):
    print(f"Available keys: {list(data_dict.keys())}")

    # 2. Access the tensor using its key
    # --> Look at the output above and replace 'x' with the correct key for your tensor.
    # --> Common keys are 'latent', 'data', or 'x'.
    tensor_key = 'z' 
    if tensor_key in data_dict:
        latent_tensor = data_dict[tensor_key]

        print(f"--- Statistics for tensor with key '{tensor_key}' ---")
        if hasattr(latent_tensor, 'shape'):
            print("Shape:", latent_tensor.shape)
            print("Mean: ", latent_tensor.mean().item())
            print("Std:  ", latent_tensor.std().item())
            print("Min:  ", latent_tensor.min().item())
            print("Max:  ", latent_tensor.max().item())
        else:
            print(f"The value associated with key '{tensor_key}' is not a tensor. It is a {type(latent_tensor)}.")

    else:
        print(f"Error: Key '{tensor_key}' not found. Please replace it with one of the available keys: {list(data_dict.keys())}")
elif torch.is_tensor(data_dict):
    print("--- Statistics for the loaded tensor ---")
    latent_tensor = data_dict
    print("Shape:", latent_tensor.shape)
    print("Mean: ", latent_tensor.mean().item())
    print("Std:  ", latent_tensor.std().item())
    print("Min:  ", latent_tensor.min().item())
    print("Max:  ", latent_tensor.max().item())
else:
    print(f"Loaded file is not a dictionary or a tensor. It is a {type(data_dict)}. Cannot inspect.")
