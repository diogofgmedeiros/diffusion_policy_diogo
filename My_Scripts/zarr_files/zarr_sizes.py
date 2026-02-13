import zarr

path = "/home/medeiros/Desktop/Projects/diffusion_policy/data/training/block_pushing/multimodal_push_seed.zarr"
z = zarr.open(path, mode="r")

print("\n")
print("\033[1m----- data -----\033[0m")
for name, arr in z["data"].arrays():
    print(f"{name} -> shape={arr.shape}, dtype={arr.dtype}")
print("\n")
print("\033[1m----- meta -----\033[0m")

for k in z["meta"].keys():
    arr = z["meta"][k]
    print(f"{k} -> shape={arr.shape}, dtype={arr.dtype}")

print("\n")
