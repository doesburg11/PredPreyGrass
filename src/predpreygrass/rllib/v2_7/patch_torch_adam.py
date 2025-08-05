# patch_torch_adam.py
import torch.optim

if not hasattr(torch.optim, "_patched_foreach"):
    _original_adam = torch.optim.Adam

    class PatchedAdam(_original_adam):
        def __init__(self, *args, **kwargs):
            kwargs["foreach"] = False
            super().__init__(*args, **kwargs)

    torch.optim.Adam = PatchedAdam
    torch.optim._patched_foreach = True
