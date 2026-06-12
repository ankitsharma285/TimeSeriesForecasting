from .naive import NaivePersistence, RepeatLastWindow
from .Linearv2 import Model as LinearV2Model
from .DLinear import Model as DLinearModel

# Expose a clean registry mapping
MODEL_REGISTRY = {
    "naive_persistence": NaivePersistence,
    "window_repeat": RepeatLastWindow,
    "linear_v2": LinearV2Model,
    "dlinear": DLinearModel
}

def get_model(model_name: str, seq_len: int, pred_len: int, enc_in: int, individual: bool = True):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model architecture: {model_name}. Choose from {list(MODEL_REGISTRY.keys())}")
        
    model_class = MODEL_REGISTRY[model_name]
    
    # Statistical baselines don't accept the 'individual' flag
    if "naive" in model_name or "window" in model_name:
        return model_class(seq_len=seq_len, pred_len=pred_len, enc_in=enc_in)
    
    # Neural models take the full configuration block
    return model_class(seq_len=seq_len, pred_len=pred_len, enc_in=enc_in, individual=individual)