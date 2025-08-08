import torch

checkpoint_path = 'C://Users//maria//Desktop//deepfake//ramo1//model_kd//teacher.pth'  # Percorso al file scaricato

checkpoint = torch.load(checkpoint_path, map_location='cpu')

# Se è un dizionario salvato da PySlowFast / AltFreezing, vedremo queste chiavi:
print("Chiavi principali:", checkpoint.keys())

# Se c'è una chiave 'model_state', 'state_dict', o simile
if 'model_state' in checkpoint:
    state = checkpoint['model_state']
elif 'state_dict' in checkpoint:
    state = checkpoint['state_dict']
else:
    state = checkpoint

# Stampa alcune delle chiavi dei pesi per capire il backbone
print("\nEsempio di layer contenuti:")
for k in list(state.keys())[:10]:
    print(k)
