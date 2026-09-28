import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# reading process testing via file path
route = "/Users/angelorlandopena/Documents/GenoBit/Investigación/NueroImaging/OASIS/disc1/OAS1_0001_MR1/PROCESSED/MPRAGE/T88_111/OAS1_0001_MR1_mpr_n4_anon_111_t88_gfc.img"

# loads the .img file 
img = nib.load(route)
# extracts the 3D voxel data as a NumPy array
data = img.get_fdata()

# show metadata
# print("Dimentions: ", data.shape)
# print("Data type: ", data.dtype)
# print("Minimum value: ", np.min(data))
# print("Maximum value: ", np.max(data))


# RESULTS Meta data test
# test approved
# metadata printed on terminal


# INVESTIGAR POR QUÉ NO FUNCIONA
# corte axial central

# slice_index = data.shape[2] // 2  
# plt.imshow(data[:, :, slice_index].T, cmap="gray", origin="lower")
# plt.title("Corte axial central")
# plt.axis("off")
# plt.show()


data = img.get_fdata()
print("Original shape: ", data.shape)

data = np.squeeze(data)
print("New shape:", data.shape)

# Coronal Section
slice_index = data.shape[2] // 2
plt.imshow(data[:, slice_index, :].T, cmap="gray", origin="lower")
plt.title("Corte Coronal")

# Axial Section
# slice_index = data.shape[1] // 2
# plt.imshow(data[:, :, slice_index].T, cmap="gray", origin="lower")
# plt.title("Corte Axial")

# Sagital Section
# slice_index = data.shape[0] // 2
# plt.imshow(data[slice_index, :, :].T, cmap="gray", origin="lower")
# plt.title("Corte Sagital")

#testing another way of using slice index on coordinates
#this index corresponds to axial slice
# slice_index = data.shape[0] // 2
#using this index as the sagital section coordinate
# plt.imshow(data[slice_index, :, :].T, cmap="gray", origin="lower")

plt.axis("off")
plt.show()

