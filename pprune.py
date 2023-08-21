import torch
import torch.nn.utils.prune as prune
import torchvision.models as models
import torchvision.models as m
import torch.nn as nn

# Definisci il tuo modello MobileNetV2
#model = m.mobilenet_v2(weights=m.MobileNet_V2_Weights.IMAGENET1K_V2)
#model.load_state_dict(torch.load("./modelli/mobilenetv2.pt"))
model=torch.load("./modelli/mobilenetv2.pth")
num_params_before = sum(p.numel() for p in model.parameters())
print(f"Numero di parametri del modello prima del pruning: {num_params_before}")
# Applica il pruning al modello
prune_rate = 0.9 # tasso di pruning, ossia la percentuale di connessioni che devono essere eliminate
for module in model.modules():
    if isinstance(module, torch.nn.Conv2d):
        prune.l1_unstructured(module, name='weight', amount=prune_rate)
        if prune.is_pruned(module):
            print(f"All parameters in module are pruned.")
        else:
            print(f"Some parameters in modulev are not pruned.")
            #print(module.weight.data)

torch.nn.utils.prune.remove(model, 'weight')
# Verifica se il modulo weightConv2d del primo layer è stato prunato
first_conv_layer = next(iter(model.features._modules.items()))[1][0]
is_pruned = hasattr(first_conv_layer, 'weight_orig') # Restituisce True se il modulo è stato prunato, False altrimenti

print(is_pruned)
num_params_after=torch.sum(torch.tensor([torch.numel(p) for p in model.parameters() if p.requires_grad]))
print(f"Numero di parametri del modello dopo il pruning: {num_params_after}")
