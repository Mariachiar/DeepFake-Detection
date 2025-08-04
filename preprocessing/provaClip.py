import numpy as np
import matplotlib.pyplot as plt

# Carica la clip
clip = np.load("C://Users//maria//Desktop//deepfake//datasets//processed_dataset//01_02__meeting_serious__YVGY8LOK//track_1//clip_00001//images.npy")  # shape attesa: (8, 224, 224, 3)

print("Forma del file:", clip.shape)  # Controllo: deve essere (8, 224, 224, 3)

# Mostra tutti gli 8 frame della clip
for i in range(clip.shape[0]):
    plt.subplot(2, 4, i + 1)
    plt.imshow(clip[i])
    plt.axis('off')
    plt.title(f"Frame {i+1}")

plt.suptitle("Clip visualizzata (track2_frame64.npy)")
plt.tight_layout()
plt.show()
