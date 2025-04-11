
#### Neural Network Optimizations and Embeddings


This repository contains custom implementations of frequency-adaptive optimization algorithms n-dimensional rotary positional embeddings and attentions for transformers and tranformer-like architectures that are nlp/asr focused. And a few other things. These naturally lend themselves to vision and multimodal. Everything here is in a forever wip state and tends to be buggy. 


``` python
    def create_encoder_layers(num_layers=10, dims=512):
        variations = {
            "activation": [nn.ReLU(), nn.GELU(), nn.SELU(), nn.LeakyReLU()],
            "dropout": lambda: np.random.uniform(0.1, 0.3),
            "head": lambda: np.random.choice([4, 8, 12]),
            "norm_type": ["layernorm", "rmsnorm"]
        }

        layers = nn.ModuleList([
            Residual(
                dims=dims,
                head=variations["head"](),
                layer_idx=i,
                act=np.random.choice(variations["activation"]),
                decoder=False,
                cross_attention=False
            ) for i in range(num_layers)
        ])
        return layers

```

The full code for each snippet can found somewhere on this github. These are ideas. More often than not they are less than entirely pratical but every once in a blue moon something kind of works so I post some of those here.  

Frequency-Adaptive Momentum (FAM) Optimizer


```python

class FAMOptimizer(torch.optim.Optimizer):
    """
    Frequency-Adaptive Momentum optimizer with parameter-specific handlers.
    
    Args:
        ... (existing parameters)
        debug (bool, optional): Whether to collect debug information (default: False)
        debug_dir (str, optional): Directory to save debug info (default: './fam_debug')
        debug_interval (int, optional): Steps between debug dumps (default: 1000)
    """
    def __init__(self, params, lr=1e-3, alpha=0.9, beta=0.99, eps=1e-8,
                 weight_decay=0, n_bands=8, fam_start_step=100,
                 layer_boost=True, min_size=256, debug=False,
                 debug_dir='./fam_debug', debug_interval=1000):
        defaults = dict(lr=lr, alpha=alpha, beta=beta, eps=eps,
                       weight_decay=weight_decay, n_bands=n_bands,
                       fam_start_step=fam_start_step, 
                       layer_boost=layer_boost, min_size=min_size)
        self.debug = debug
        self.debug_info = {} if debug else None
        
        self.debug_dir = debug_dir
        self.debug_interval = debug_interval
        self.last_dump_step = 0
        
        if debug and debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
            self.debug_file = os.path.join(
                debug_dir, 
                f"fam_debug_{datetime.now().strftime('%m%d_%H%M%S')}.json"
            )
            with open(self.debug_file, 'w') as f:
                json.dump({
                    "optimizer": "FAMOptimizer",
                    "settings": {
                        "lr": lr,
                        "alpha": alpha,
                        "beta": beta,
                        "n_bands": n_bands,
                        "fam_start_step": fam_start_step,
                    },
                    "parameters": {},
                    "steps_recorded": []
                }, f, indent=2)
            print(f"FAM debug info will be saved to {self.debug_file}")
        
        self.handlers = {
            "default": FrequencyHandler(),
            "conv": ConvFrequencyHandler(),
            "attention": AttentionFrequencyHandler(),
            "embedding": EmbeddingFrequencyHandler()
        }
        
        param_groups = self._add_handlers_to_groups(params)
        super(FAMOptimizer, self).__init__(params=param_groups, defaults=defaults)
        
        print(f"FAM Optimizer initialized with parameter-specific handlers:") 
        print(f"  lr={lr}, alpha={alpha}, beta={beta}, n_bands={n_bands}")
        print(f"  fam_start_step={fam_start_step}, min_size={min_size}")
    
    def _add_handlers_to_groups(self, params):
        """Add appropriate handlers to parameter groups based on type"""
        if isinstance(params, list) and all(isinstance(pg, dict) for pg in params):
            for pg in params:
                if 'handler' not in pg:
                    if any('conv' in name.lower() for name in pg.get('names', [])):
                        pg['handler'] = 'conv'
                    elif any(name in name.lower() for name in pg.get('names', []) 
                             for name in ['attention', 'mha', 'self_attn']):
                        pg['handler'] = 'attention'
                    elif any(name in name.lower() for name in pg.get('names', [])
                             for name in ['embed', 'token']):
                        pg['handler'] = 'embedding'
                    else:
                        pg['handler'] = 'default'
            return params
        else:
            return [{'params': params, 'handler': 'default'}]
    
    def get_handler(self, group):
        """Get the appropriate frequency handler for the parameter group"""
        handler_name = group.get('handler', 'default')
        return self.handlers[handler_name]
    
    def dump_debug_info(self, force=False):
        """Save the current debug information to file"""
        if not self.debug or not hasattr(self, 'debug_file'):
            return
        
        current_step = max([self.state[p]['step'] for p in self.state], default=0)
        
        if force or (current_step - self.last_dump_step >= self.debug_interval):
            try:
                with open(self.debug_file, 'r') as f:
                    debug_data = json.load(f)
                
                debug_data["steps_recorded"].append(current_step)
                
                for param_name, param_info in self.debug_info.items():
                    if param_name not in debug_data["parameters"]:
                        debug_data["parameters"][param_name] = {
                            "handler": param_info.get('handler', 'default'),
                            "steps": [],
                            "bands": [],
                            "alpha": []
                        }
                    
                    last_recorded = len(debug_data["parameters"][param_name]["steps"])
                    if last_recorded < len(param_info['steps']):
                        debug_data["parameters"][param_name]["steps"].extend(param_info['steps'][last_recorded:])
                        debug_data["parameters"][param_name]["bands"].extend(param_info['bands'][last_recorded:])
                        debug_data["parameters"][param_name]["alpha"].extend(param_info['alpha'][last_recorded:])
                
                with open(self.debug_file, 'w') as f:
                    json.dump(debug_data, f)
                
                self.last_dump_step = current_step
                
                for param_info in self.debug_info.values():
                    param_info['steps'] = param_info['steps'][-10:]
                    param_info['bands'] = param_info['bands'][-10:]
                    param_info['alpha'] = param_info['alpha'][-10:]
                    
            except Exception as e:
                print(f"Error dumping FAM debug info: {e}")
    
    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p_idx, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('FAMOptimizer does not support sparse gradients')
                
                state = self.state[p]
                
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['freq_history'] = {}
                    state['param_name'] = f"param_{p_idx}"
                
                state['step'] += 1
                
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])
                
                exp_avg = state['exp_avg']
                alpha = group['alpha']
                beta = group['beta']
                lr = group['lr']
                n_bands = group['n_bands']
                
                handler = self.get_handler(group)
                
                should_apply_fam = (
                    state['step'] > group['fam_start_step'] and
                    p.numel() > group['min_size']
                )
                
                if should_apply_fam:
                    try:
                        if p.numel() > 10000:
                            if p.dim() > 1:
                                row_indices = torch.randperm(p.size(0))[:min(p.size(0), 64)]
                                col_indices = torch.randperm(p.size(1))[:min(p.size(1), 64)]
                                grad_sample = grad[row_indices][:, col_indices].flatten()
                            else:
                                sample_idx = torch.randperm(p.numel())[:1000]
                                grad_sample = grad.flatten()[sample_idx]
                        else:
                            grad_sample = grad.flatten()
                        
                        band_powers = handler.analyze(grad_sample, n_bands, group['eps'])
                        
                        if state['step'] <= 10 and p_idx == 0:
                            print(f"Step {state['step']}: Found {len(band_powers)} frequency bands")
                            print(f"Band powers: {[f'{v:.4f}' for v in band_powers]}")
                        
                        for i, power in enumerate(band_powers):
                            band_key = f'band_{i}'
                            if band_key not in state['freq_history']:
                                state['freq_history'][band_key] = power
                            else:
                                state['freq_history'][band_key] = (
                                    beta * state['freq_history'][band_key] +
                                    (1-beta) * power
                                )
                        
                        band_values = [state['freq_history'].get(f'band_{i}', 0) 
                                      for i in range(n_bands)]
                        
                        effective_alpha = handler.get_adaptive_momentum(band_values, alpha)
                        
                        if self.debug:
                            param_name = state['param_name']
                            if param_name not in self.debug_info:
                                self.debug_info[param_name] = {
                                    'steps': [], 
                                    'bands': [], 
                                    'handler': group.get('handler', 'default'),
                                    'alpha': []
                                }
                            
                            if state['step'] % 10 == 0:
                                self.debug_info[param_name]['steps'].append(state['step'])
                                self.debug_info[param_name]['bands'].append(band_values)
                                self.debug_info[param_name]['alpha'].append(effective_alpha)
                        
                        exp_avg.mul_(effective_alpha).add_(grad, alpha=1-effective_alpha)
                    except Exception as e:
                        import traceback
                        print(f"Error in FAM processing for parameter {p_idx}:")
                        print(f"Error type: {type(e).__name__}")
                        print(f"Error message: {e}")
                        print(f"Parameter shape: {p.shape}, numel: {p.numel()}")
                        print(traceback.format_exc())
                        exp_avg.mul_(alpha).add_(grad, alpha=1-alpha)
                else:
                    exp_avg.mul_(alpha).add_(grad, alpha=1-alpha)
                
                p.add_(exp_avg, alpha=-lr)
        
        if self.debug:
            self.dump_debug_info()
        
        return loss
    
    def __del__(self):
        """Clean up and final debug dump when optimizer is destroyed"""
        if self.debug:
            self.dump_debug_info(force=True)


 ```
 
 1. Gradient Frequency Analysis
 The FAM optimizer analyzes gradient frequency spectra to dynamically adjust optimization parameters. This addresses the challenge that different parameter types (attention, embeddings, convolutions) require different update strategies.
 
 2. Parameter-Specific Handlers
```python
class FrequencyHandler:
    """Base class for parameter-specific frequency analysis functions"""
    
    def analyze(self, grad_sample, n_bands, eps=1e-8):
        """Default frequency analysis implementation"""
        freq_repr = torch.fft.rfft(grad_sample.float())
        freq_power = torch.abs(freq_repr)
        
        if freq_power.sum() > 0:
            freq_power = freq_power / (freq_power.sum() + eps)
        
        # Divide into bands
        band_size = freq_power.shape[0] // n_bands
        if band_size <= 0:
            return [0.0] * n_bands
            
        band_powers = []
        for i in range(n_bands):
            start_idx = i * band_size
            end_idx = min((i+1) * band_size, freq_power.shape[0])
            if start_idx < end_idx:
                band_power = freq_power[start_idx:end_idx].sum().item()
                band_powers.append(band_power)
            else:
                band_powers.append(0.0)
                
        return band_powers
    
    def get_adaptive_momentum(self, band_values, base_alpha):
        """Default adaptive momentum calculation"""
        n_bands = len(band_values)
        high_freq_activity = sum(band_values[n_bands//2:])
        
        if high_freq_activity > 0.3:
            return min(0.95, base_alpha + 0.05)
        return base_alpha


class ConvFrequencyHandler(FrequencyHandler):
    """Specialized handler for convolutional layers"""
    
    def analyze(self, grad_sample, n_bands, eps=1e-8):
        # More precise sampling for convolutional layers
        freq_repr = torch.fft.rfft(grad_sample.float())
        freq_power = torch.abs(freq_repr)
        
        if freq_power.sum() > 0:
            freq_power = freq_power / (freq_power.sum() + eps)
        
        # Use logarithmically spaced bands for convolution layers
        # to better capture both low and high frequency patterns
        band_powers = []
        total_freqs = freq_power.shape[0]
        
        for i in range(n_bands):
            # Log-spaced indices
            start_idx = int((total_freqs ** (i/n_bands)) - 1)
            end_idx = int((total_freqs ** ((i+1)/n_bands)) - 1)
            start_idx = max(0, start_idx)
            end_idx = min(end_idx, total_freqs)
            
            if start_idx < end_idx:
                band_power = freq_power[start_idx:end_idx].sum().item()
                band_powers.append(band_power)
            else:
                band_powers.append(0.0)
                
        return band_powers
    
    def get_adaptive_momentum(self, band_values, base_alpha):
        """Convolutional layers benefit from more smoothing in mid-frequencies"""
        n_bands = len(band_values)
        
        # Calculate band distribution 
        mid_freq_activity = sum(band_values[n_bands//4:(3*n_bands)//4])
        high_freq_activity = sum(band_values[(3*n_bands)//4:])
        
        # Increase momentum more for mid-frequency noise that often appears in conv layers
        if mid_freq_activity > 0.4:
            return min(0.97, base_alpha + 0.07)
        elif high_freq_activity > 0.3:
            return min(0.95, base_alpha + 0.05)
        return base_alpha

class AttentionFrequencyHandler(FrequencyHandler):
    """Specialized handler for attention layers"""
    
    def analyze(self, grad_sample, n_bands, eps=1e-8):
        # Standard frequency analysis but with more bands in higher frequencies
        freq_repr = torch.fft.rfft(grad_sample.float())
        freq_power = torch.abs(freq_repr)
        
        if freq_power.sum() > 0:
            freq_power = freq_power / (freq_power.sum() + eps)
        
        # Attention layers often have important high-frequency patterns
        # Use more bands in high frequencies
        band_powers = []
        half_bands = n_bands // 2
        
        # Low frequency bands (first half)
        low_band_size = (freq_power.shape[0] // 2) // half_bands
        for i in range(half_bands):
            start_idx = i * low_band_size
            end_idx = min((i+1) * low_band_size, freq_power.shape[0] // 2)
            if start_idx < end_idx:
                band_power = freq_power[start_idx:end_idx].sum().item()
                band_powers.append(band_power)
            else:
                band_powers.append(0.0)
                
        # High frequency bands (second half with more detail)
        high_band_size = (freq_power.shape[0] - (freq_power.shape[0] // 2)) // (n_bands - half_bands)
        for i in range(half_bands, n_bands):
            start_idx = (freq_power.shape[0] // 2) + (i - half_bands) * high_band_size
            end_idx = min((freq_power.shape[0] // 2) + (i - half_bands + 1) * high_band_size, freq_power.shape[0])
            if start_idx < end_idx:
                band_power = freq_power[start_idx:end_idx].sum().item()
                band_powers.append(band_power)
            else:
                band_powers.append(0.0)
                
        return band_powers
    
    def get_adaptive_momentum(self, band_values, base_alpha):
        """Custom adaptive momentum for attention layers"""
        n_bands = len(band_values)
        
        # Get band with maximum energy
        max_band_idx = np.argmax(band_values)
        
        # Attention matrices often benefit from lower momentum for low frequencies
        if max_band_idx < n_bands // 4:
            # Dominant low frequency - less smoothing
            return max(0.85, base_alpha - 0.05)
        elif max_band_idx > 3*n_bands // 4:
            # Dominant high frequency - more smoothing
            return min(0.98, base_alpha + 0.08)
        return base_alpha


class EmbeddingFrequencyHandler(FrequencyHandler):
    """Specialized handler for embedding layers"""
    
    def get_adaptive_momentum(self, band_values, base_alpha):
        """Embeddings often benefit from very stable updates"""
        n_bands = len(band_values)
        
        # More aggressive smoothing for high-frequency components in embeddings
        high_freq_activity = sum(band_values[(3*n_bands)//4:])
        if high_freq_activity > 0.2:  # Lower threshold for embeddings
            return min(0.98, base_alpha + 0.08)
        return base_alpha

```         
 Each parameter type gets a specialized frequency handler that understands the unique update patterns required:
 - Uses logarithmically spaced bands to better capture convolution filter patterns
 - Emphasizes high-frequency components crucial for attention matrices
 - Applies more smoothing to stabilize embedding updates
 
 3. Adaptive Momentum Calculation
 ```python    
        def get_adaptive_momentum(self, band_values, base_alpha):
            """Dynamically adjust momentum based on frequency distribution"""
            n_bands = len(band_values)
            high_freq_activity = sum(band_values[n_bands//2:])
        
        if high_freq_activity > 0.3:
            return min(0.95, base_alpha + 0.05)
        return base_alpha
```
  Dynamically adjusts momentum coefficients based on gradient frequency characteristics:
 - Higher momentum for high-frequency noise (smoother updates)
 - Lower momentum for meaningful low-frequency components (faster learning)
 - Fall back to regular Maxfactor
 
 4. Debug and Monitoring Tools
 
 Includes debug tools to track frequency band distribution across training, helping identify optimization challenges. (this is mostly for my sanity)

Mini version and schedulers (scheduler used for regular fam too)

```python

class FAMScheduler(_LRScheduler):
    """
    Scheduler with linear warmup followed by cosine annealing.
    
    Args:
        optimizer: Wrapped optimizer
        warmup_epochs: Number of epochs for the linear warmup
        max_epochs: Total number of epochs
        warmup_start_lr: Initial learning rate for warmup
        eta_min: Minimum learning rate after cosine annealing
    """
    def __init__(self, optimizer, warmup_epochs, max_epochs, warmup_start_lr=1e-8, eta_min=1e-8, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        super(FAMScheduler, self).__init__(optimizer, last_epoch)
        
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / self.warmup_epochs
            return [self.warmup_start_lr + (base_lr - self.warmup_start_lr) * alpha for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            return [self.eta_min + (base_lr - self.eta_min) * 
                   (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / 
                                (self.max_epochs - self.warmup_epochs))) / 2
                   for base_lr in self.base_lrs]

class SimpleFAM(torch.optim.Optimizer):
    """
    Simplified Frequency-Adaptive Momentum optimizer
    
    A lightweight implementation that focuses on the core concepts
    without complex debugging or parameter-specific handlers.
    """
    def __init__(self, params, lr=0.001, alpha=0.9, beta=0.99):
        defaults = dict(lr=lr, alpha=alpha, beta=beta)
        super(SimpleFAM, self).__init__(params, defaults)
        print(f"SimpleFAM initialized with lr={lr}, alpha={alpha}")
        
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                
                state['step'] += 1
                exp_avg = state['exp_avg']
                alpha = group['alpha']
                
                # Only apply FAM to large tensors
                if p.numel() > 1000 and state['step'] > 100:
                    # Simple frequency analysis
                    grad_sample = p.grad.flatten()[:min(1000, p.numel())]
                    freq = torch.fft.rfft(grad_sample.float())
                    power = torch.abs(freq)
                    
                    # Calculate high vs low frequency ratio
                    half = power.shape[0] // 2
                    high_ratio = power[half:].sum() / (power.sum() + 1e-8)
                    
                    # Adjust momentum based on frequency
                    effective_alpha = min(0.98, alpha + 0.05 * high_ratio)
                    exp_avg.mul_(effective_alpha).add_(p.grad, alpha=1-effective_alpha)
                else:
                    # Standard momentum
                    exp_avg.mul_(alpha).add_(p.grad, alpha=1-alpha)
                
                # Update weights
                p.add_(exp_avg, alpha=-group['lr'])
        
        return loss

# Also add FAMScheduler2 if it doesn't exist
class FAMScheduler2(torch.optim.lr_scheduler._LRScheduler):
    """
    Step-based learning rate scheduler for FAM optimizer
    with warmup and cosine annealing.
    """
    def __init__(self, optimizer, warmup_steps=1000, total_steps=100000, 
                 decay_start_step=None, warmup_start_lr=1e-6, eta_min=1e-6, 
                 last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.decay_start_step = decay_start_step if decay_start_step is not None else warmup_steps
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        super(FAMScheduler2, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup phase
            alpha = self.last_epoch / self.warmup_steps
            return [self.warmup_start_lr + (base_lr - self.warmup_start_lr) * alpha 
                    for base_lr in self.base_lrs]
        
        elif self.last_epoch < self.decay_start_step:
            # Optional plateau phase (constant LR between warmup and decay)
            return self.base_lrs
        
        else:
            # Cosine annealing decay phase
            return [self.eta_min + (base_lr - self.eta_min) * 
                   (1 + math.cos(math.pi * (self.last_epoch - self.decay_start_step) / 
                                (self.total_steps - self.decay_start_step))) / 2 + 1e-8
                   for base_lr in self.base_lrs]
```
 
### nDimensional Rotary Embedding (Givens-Quaternion with regular RoPE fall back)

> - Incorporates quaternion rotations (`q_rotation`), enabling transformations in three-dimensional space by utilizing normalized unit vectors and angle-based adjustments with added compatibility for quaternion-based rotational enhancements in high-dimensional contexts. 4d, 6d, 8d, ... etc.

> - Maps high-dimensional embeddings to lower dimensions before rotation and reconstructs them afterward. This process uses singular value decomposition (SVD) for orthogonal initialization, aimed at maintaining numerical stability. Frequency-based encoding (`freqs`) supports learned parameters, allowing dynamic representation adjustments.
 
 ```python

class RotaryEmbedding(nn.Module):

    def __init__( self, dim, theta = 10000, num_freqs = 1, learned_freq = True, theta_rescale_factor = 1., 
                 use_quaternion = False, rot_scale = 1.0, rot_count = 1, use_projection = False, proj_dim = 3, 
                 proj_scale = 0.1, ): 
        super().__init__()
        theta *= theta_rescale_factor ** (dim / (dim - 2))
        self.freqs = nn.Parameter(torch.arange(0, num_freqs) * (2 * math.pi / theta), requires_grad=learned_freq)
        self.register_buffer('dummy', torch.tensor(0), persistent=False)
        self.use_quaternion = use_quaternion
        self.use_projection = use_projection
        self.proj_dim = proj_dim
        self.proj_scale = proj_scale
        
        if use_quaternion:
            self.dparam = nn.Parameter(torch.zeros(1))
            self.rscale = rot_scale
            self.rot = rot_count
            self.tscale = 1.0
            pairs = []
            for i in range(0, dim-1, 2):
                pairs.append(torch.tensor([i, i+1]))
            self.pairs = nn.Parameter(torch.stack(pairs), requires_grad=False)
            self.thetas = nn.Parameter(torch.ones(len(self.pairs)) * (2 * math.pi / len(self.pairs)), 
                                      requires_grad=False)
            if use_projection:
                self.proj_down = None
                self.proj_up = None

    @property
    def device(self):
        return self.dummy.device

    def q_rotation(self, x, theta, u, v=None):
        eps = 1e-8
        u_norm = torch.norm(u, p=2)
        u = u / (u_norm + eps)
        w = torch.cos(theta / 2)
        vec = torch.sin(theta / 2) * u
        x_shape = x.shape
        x = x.reshape(-1, 3)
        uv_cross = torch.cross(u.unsqueeze(0), x)
        uuv_cross = torch.cross(u.unsqueeze(0), uv_cross)
        x_rot = x + torch.clamp(2 * (w * uv_cross + uuv_cross), min=-10.0, max=10.0)
        return x_rot.reshape(*x_shape)

    def rotation_matrix(self, dims, i, j, theta):
        G = torch.eye(dims, device=theta.device)
        c, s = torch.cos(theta), torch.sin(theta)
        G[i, i], G[j, j] = c, c
        G[i, j], G[j, i] = -s, s
        if dims == 3:
            u = torch.eye(dims, device=theta.device)[i]
            v = torch.eye(dims, device=theta.device)[j]
            if theta < 0: 
                Q = self.q_rotation(torch.eye(dims, device=theta.device), theta=abs(theta), u=u, v=v)
            else:
                Q = self.q_rotation(torch.eye(dims, device=theta.device), theta=theta, u=u, v=v)
            G = (G + Q) / 2
        return G

    def rotations(self, x):
        direction = torch.sigmoid(self.dparam) * 2 - 1
        rotate = int(round(self.rscale * self.rot))
        head_dim = x.shape[-1]
        for k in range(min(rotate, len(self.pairs))):
            i, j = self.pairs[k].long()
            if i >= head_dim or j >= head_dim:
                continue
            theta = direction * self.thetas[k] * self.tscale
            G = self.rotation_matrix(dims=head_dim, i=i.item(), j=j.item(), theta=theta)
            x_shape = x.shape
            x = x.reshape(-1, head_dim)
            x = x @ G
            x = x.reshape(*x_shape)
        return x

    def _ensure_projection(self, x):
        if self.proj_down is None or self.proj_down.weight.device != x.device:
            head_dim = x.shape[-1] 
            self.proj_down = Linear(head_dim, self.proj_dim, bias=False).to(x.device)
            self.proj_up = Linear(self.proj_dim, head_dim, bias=False).to(x.device)
            with torch.no_grad():
                nn.init.orthogonal_(self.proj_down.weight, gain=self.proj_scale)
                nn.init.orthogonal_(self.proj_up.weight, gain=self.proj_scale)
                U, S, V = torch.svd(self.proj_down.weight)
                S_inv = 1.0 / (S + 1e-6) 
                S_inv = torch.clamp(S_inv, max=10.0)
                pseudo_inv = V @ torch.diag(S_inv) @ U.t()
                self.proj_up.weight.copy_(pseudo_inv * self.proj_scale)

    def project_and_rotate(self, x):
        orig_shape = x.shape
        x_flat = x.reshape(-1, x.shape[-1])
        with torch.no_grad():
            x_norm = torch.norm(x_flat, dim=1, keepdim=True)
            if torch.max(x_norm) > 1e3:
                x_flat = x_flat * (1e3 / torch.max(x_norm))
        if x.shape[-1] > 3 and self.use_projection:
            self._ensure_projection(x)
            x_3d = self.proj_down(x_flat)
            if torch.isnan(x_3d).any():
                return x.reshape(*orig_shape)
            x_3d_rot = self.rotations(x_3d)
            if torch.isnan(x_3d_rot).any():
                x_rot = self.proj_up(x_3d)
            else:
                x_rot = self.proj_up(x_3d_rot)
            alpha = 0.9
            x_rot = alpha * x_rot + (1-alpha) * x_flat
            if torch.isnan(x_rot).any():
                return x.reshape(*orig_shape)
        else:
            x_rot = self.rotations(x_flat)
        return x_rot.reshape(*orig_shape)

    def apply_rotary(self, freqs, t, start_index=0, scale=1., seq_dim=-2, freqs_seq_dim=None):
        dtype = t.dtype
        
        def _exists(val):
            return val is not None
        
        def _slice_at_dim(tensor, dim_slice, dim):
            dim += (tensor.ndim if dim < 0 else 0)
            colons = [slice(None)] * tensor.ndim
            colons[dim] = dim_slice
            return tensor[tuple(colons)]
        
        def _rotate_half(x):
            x = rearrange(x, '... (d r) -> ... d r', r=2)
            x1, x2 = x.unbind(dim=-1)
            x = torch.stack((-x2, x1), dim=-1)
            return rearrange(x, '... d r -> ... (d r)')
        
        if not _exists(freqs_seq_dim):
            if freqs.ndim == 2 or t.ndim == 3:
                freqs_seq_dim = 0
                
        if t.ndim == 3 or _exists(freqs_seq_dim):
            ctx = t.shape[seq_dim]
            freqs = _slice_at_dim(freqs, slice(-ctx, None), dim=freqs_seq_dim)
        rot_dim = freqs.shape[-1]
        end_index = start_index + rot_dim
        
        assert rot_dim <= t.shape[-1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'
        t_left = t[..., :start_index]
        t_middle = t[..., start_index:end_index]
        t_right = t[..., end_index:]
        t_transformed = (t_middle * freqs.cos() * scale) + (_rotate_half(t_middle) * freqs.sin() * scale)
        out = torch.cat((t_left, t_transformed, t_right), dim=-1)
        return out.type(dtype)

    def rotate_qk(self, t, seq_dim=None, offset=0, scale=None):
        if self.use_quaternion:
            if self.use_projection and t.shape[-1] > 3:
                return self.project_and_rotate(t)
            else:
                return self.rotations(t)
        else:
            ctx = t.shape[2]
            device, dtype = t.device, t.dtype
            seq = torch.arange(ctx, device=device, dtype=dtype) + offset
            freqs = self.forward(seq)
            scale = scale if scale is not None else 1.0
            return self.apply_rotary(freqs, t, scale=scale, seq_dim=2)
    
    def learned_rotations(self, rotations, t, start_index = 0, freq_ranges = None):
        if exists(freq_ranges):
            rotations = einsum('..., f -> ... f', rotations, freq_ranges)
            rotations = rearrange(rotations, '... r f -> ... (r f)')
        rotations = repeat(rotations, '... n -> ... (n r)', r = 2)
        return self.apply_rotary(rotations, t, start_index = start_index)

    def forward(self, t):
        freqs = self.freqs
        freqs = torch.einsum('..., f -> ... f', t.type(freqs.dtype), freqs)
        freqs = torch.repeat_interleave(freqs, 2, dim=-1)
        return freqs

### mini version
class CompactRotation:
    def __init__(self, dim, rot_pairs=None, rot_scale=1.0, rot_count=1):
        self.scale = rot_scale
        self.count = rot_count

        if rot_pairs is None:
            pairs = []
            for i in range(0, dim-1, 2):
                pairs.append((i, i+1))
            self.pairs = pairs
        else:
            self.pairs = rot_pairs
            
        self.thetas = [2 * math.pi / len(self.pairs)] * len(self.pairs)
        self.direction = 1.0  # Fixed direction instead of learned
        
    def __call__(self, x):
        """Apply rotations to input tensor"""
        return self.rotate(x)
        
    def rotate(self, x):
        rotate_steps = min(int(round(self.scale * self.count)), len(self.pairs))
        head_dim = x.shape[-1]
        
        for k in range(rotate_steps):
            i, j = self.pairs[k]
            if i >= head_dim or j >= head_dim:
                continue
                
            # Create rotation matrix
            theta = self.direction * self.thetas[k]
            device = x.device
            G = torch.eye(head_dim, device=device)
            c, s = torch.cos(theta), torch.sin(theta)
            G[i, i], G[j, j] = c, c
            G[i, j], G[j, i] = -s, s
            
            # Apply rotation
            x_shape = x.shape
            x = x.reshape(-1, head_dim)
            x = x @ G
            x = x.reshape(*x_shape)
            
        return x
    
    @staticmethod
    def q_rotate(x, theta, axis_idx, dims=3):
        """Quaternion rotation in 3D space (simplified)"""
        device = x.device
        u = torch.zeros(dims, device=device)
        u[axis_idx] = 1.0
        
        x_shape = x.shape
        x = x.reshape(-1, dims)
        
        # Quaternion rotation formula
        uv_cross = torch.cross(u.unsqueeze(0), x)
        uuv_cross = torch.cross(u.unsqueeze(0), uv_cross)
        w = torch.cos(theta / 2)
        x_rot = x + 2 * (w * uv_cross + uuv_cross)
        
        return x_rot.reshape(*x_shape)
```
 1. Quaternion-Based Rotations
 ```python
        def q_rotation(self, x, theta, u, v=None):
             Quaternion rotation implementation for 3D space
            eps = 1e-8
            u_norm = torch.norm(u, p=2)
            u = u / (u_norm + eps)
            w = torch.cos(theta / 2)
            vec = torch.sin(theta / 2) * u
 ```
 Implements quaternion-based 3D rotations for more expressive positional encoding, allowing rotations in higher-dimensional space that better preserve geometric relationships.
 
 2. Dynamic Projection
 ```python
         def project_and_rotate(self, x):
             Project high-dimensional vectors to 3D, rotate, then project back
            orig_shape = x.shape
            x_flat = x.reshape(-1, x.shape[-1])
             Projection to 3D and rotation logic
```
 
 Projects high-dimensional embeddings into 3D space for rotation, then projects back to original dimensionality, enabling true geometric rotations even for high-dimensional embeddings.
 
 3. Learnable Rotation Parameters
 ```python
        def learned_rotations(self, rotations, t, start_index=0, freq_ranges=None):
            if exists(freq_ranges):
                rotations = einsum('..., f -> ... f', rotations, freq_ranges)
                rotations = rearrange(rotations, '... r f -> ... (r f)')
```
 
 Supports learnable rotation parameters, allowing the model to adapt positional embeddings to specific sequence patterns.
 
 4. Compact Implementation
 ```python
         class CompactRotations:
            def __init__(self, dim, rot_pairs=None, rot_scale=1.0, rot_count=1, learned_freq=False):
                 Lightweight implementation with full flexibility
```
 Provides a lightweight implementation option that maintains the core benefits while reducing computational overhead.
 
 Integration Architecture
 
 These components are designed to work together:
```python
    model = TransformerModel(...)
    rotary = RotaryEmbedding(dim=model.dim, use_quaternion=True, proj_dim=3)
    optimizer = FAMOptimizer(
        get_parameter_groups(model, lr=1e-4),
        debug=True
    )
    scheduler = FAMScheduler(optimizer, warmup_epochs=5, max_epochs=100)
 ```
 
 
### Attention mechanisms for neural language models and multimodal systems. 
#### (scrapped from my mess of work folders)

Different variations of your standard multihead attention block i've created for projects. Some are almost identical to one another with slight tweaks while some are wild abominations straying so far as to not be be recognized as a transformer.. 

- A diffused connected multiblock across several layers adaptly named wormattention...
- An attention that can decide not to pass information which feeds back weakening or strengthing the potential for future passes...
- A multihead that is modeled after a myelinated axon replete with nodes of Ranvier cleverly named.. myelinatedLayer :P ...
- other oddball ideas...

A lot of the blocks are connected and are not stand alone as-is but most are modular and can be used in that manner.. many can be used as drop-in replacements for a standard pytorch multiheadattention block and can inherit from that if you dont mind the overhead. 


All work.. Some well, some not so well. Some ideas are just bad ideas. 

A few example snippets:
 
 1. Adaptive Span Attention
```python
        class AdaptiveSpan(BaseAttention):
            """Attention with adaptive span size."""
            def __init__(self, dims, head, max_dist, sharpen=True, temp_scale=0.01):
                super().__init__(dims, head, max_dist)
                self.sharpen = sharpen
                self.temp_scale = temp_scale
                self.span_scale = nn.Parameter(torch.tensor(1.0))
```

 Dynamic adjustment of attention span based on content, optimizing computation while preserving modeling capacity. Effective for long sequences where full attention is unnecessary.
 
 2. MyelinatedLayer
    Bio-inspired architecture with dynamic information routing
```python

class MyelinatedLayer(BaseAttention):
    def __init__(self, dims, head, layerA=3, sparsity_threshold=0.1, max_dist=512):
        super().__init__(dims, head, max_dist)
        self.layers = nn.ModuleList()
        self.layerA = layerA
        self.sparsity_threshold = sparsity_threshold
        self.max_dist = max_dist
        
        self.node_predictors = nn.ModuleList([
            nn.Sequential(LayerNorm(dims),
                        Linear(dims, 1),
                        nn.Sigmoid()) for _ in range(layerA)])
        
        for i in range(layerA):
            self.layers.append(nn.ModuleDict({
                'ln': LayerNorm(dims),
                'gate': nn.Sequential(Linear(dims, 1), nn.Sigmoid()),
                'adapter': Linear(dims, dims) if i % 2 == 0 else None
            }))
        self.policy_net = nn.Sequential(Linear(dims, 128), nn.ReLU(), Linear(128, 3))
        self.jump_weights = nn.Parameter(torch.tensor([0.1, 0.05, 0.01]))
        
        mlp = dims * 4
        self.mlp_gate = nn.Sequential(Linear(dims, 1), nn.Sigmoid())
        self.mlp = nn.Sequential(Linear(dims, mlp), nn.GELU(), Linear(mlp, dims))
        self.mlp_ln = LayerNorm(dims)
        
        self.working_memory = nn.Parameter(torch.zeros(1, 1, dims))
        self.memory_gate = nn.Sequential(Linear(dims, 1), nn.Sigmoid())
        self.last_memory_gate_values = None

    def compute_attention(self, norm_x, mask=None, kv_cache=None, is_causal=True):
        """Compute attention with adaptive span and content-dependent updates."""
        batch, ctx = norm_x.shape[:2]
        
        q = norm_x.view(batch, ctx, self.head, -1).transpose(1, 2)
        k = norm_x.view(batch, ctx, self.head, -1).transpose(1, 2)
        v = norm_x.view(batch, ctx, self.head, -1).transpose(1, 2)

        attn_output, _ = calculate_attention(q, k, v, mask, 1.0, BaseAttention.use_sdpa, is_causal=is_causal)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, ctx, -1)
        return attn_output

    def predict_node_importance(self, x, layer_idx):
        """Dynamically determine if processing should occur at this node."""
        importance = self.node_predictors[layer_idx](x)
        return (importance > self.sparsity_threshold).float()

    def decide_jump(self, policy, jump_weights, i, layerA, x, original_x, working_memory):
        """Decide whether to jump layers based on the policy network."""
        jump_prob = policy[:, 1] if i < layerA - 1 else torch.zeros_like(policy[:, 1])
        should_jump = (torch.rand_like(jump_prob) < jump_prob).any()
        if should_jump:
            jump_length = torch.multinomial(policy, 1)[:, 0].max().item() + 1
            i_next = min(i + jump_length, layerA - 1)
            skip_weight = jump_weights[min(jump_length - 1, 2)]
            x = x + skip_weight * original_x + (1 - skip_weight) * working_memory
            return x, i_next
        return x, i + 1

    def forward(self, x, xa=None, mask=None, kv_cache=None, is_causal=True):
        batch, ctx = x.shape[:2]
        working_memory = self.working_memory.expand(batch, -1, -1)
        original_x = x
        pooled_representation = x.mean(dim=1, keepdim=False)
        policy_logits = self.policy_net(pooled_representation)
        policy = F.softmax(policy_logits, dim=-1)
        jump_history = []
        memory_gate = torch.zeros(batch, 1, 1, device=x.device)
        
        i = 0
        while i < self.layerA:
            layer = self.layers[i]
            node_importance = self.predict_node_importance(x, i)
            print(f"Node importance (Layer {i}): {node_importance}")

            if node_importance.mean() < 0.2 and i > 0:
                i += 1
                jump_history.append(i)
                continue
            norm_x = layer['ln'](x)
            attn_mask = mask * node_importance.squeeze(-1).unsqueeze(1) if mask is not None else node_importance.squeeze(-1).unsqueeze(1)
            
            if node_importance.mean() > 0.3:
                attn_output = self.compute_attention(norm_x, mask=attn_mask, kv_cache=kv_cache)
                print(f"Attention output (Layer {i}): {attn_output}")
                
                if layer['adapter'] is not None:
                    attn_output = layer['adapter'](attn_output)
                gate_value = layer['gate'](norm_x)
                x = x + gate_value * attn_output
                print(f"Updated representation (Layer {i}): {x}")
                
                memory_gate = self.memory_gate(x.mean(dim=1, keepdim=True))
                mean_x = x.mean(dim=1, keepdim=True)
                working_memory = memory_gate * working_memory + (1 - memory_gate) * mean_x
                print(f"Memory gate value: {memory_gate}")
            
            x, i = self.decide_jump(policy, self.jump_weights, i, self.layerA, x, original_x, working_memory)
            jump_history.append(i)

        self.last_memory_gate_values = memory_gate.detach().clone()
        print(f"Jump history: {jump_history}")
        mlp_importance = self.mlp_gate(x)
        mlp_output = self.mlp(self.mlp_ln(x))
        x = x + mlp_importance * mlp_output
        print(f"Final output: {x}")
        return x
```

Dynamic Force Interactions in Multi-Point Systems.. attention.

```python
class RelativeForce(nn.Module):
    def __init__(self, dims, heads, hop_levels=3):
        super().__init__()
        self.dims = dims
        self.heads = heads
        self.hop_levels = hop_levels
        
        # Force-related components
        self.force_emitter = nn.Linear(dims, dims)
        self.force_receptor = nn.Linear(dims, dims)
        self.direction_modulator = nn.Parameter(torch.randn(heads, dims))
        
        # Topological components
        self.edge_projector = nn.Linear(dims * 2, heads)
        self.hop_weights = nn.Parameter(torch.ones(hop_levels) / hop_levels)
        
        # Integration components
        self.force_topo_balance = nn.Parameter(torch.tensor(0.5))
        self.q_proj = nn.Linear(dims, dims)
        self.k_proj = nn.Linear(dims, dims)
        self.v_proj = nn.Linear(dims, dims)
        self.output = nn.Linear(dims, dims)
    
    def forward(self, x, mask=None):
        batch, seq_len = x.shape[:2]
        
        # Standard projections for values
        q = self.q_proj(x).view(batch, seq_len, self.heads, -1).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.heads, -1).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.heads, -1).transpose(1, 2)
        
        # Calculate force vectors
        emissions = self.force_emitter(x)
        receptivity = self.force_receptor(x)
        
        token_i = emissions.unsqueeze(2)  # [batch, seq, 1, dim]
        token_j = receptivity.unsqueeze(1)  # [batch, 1, seq, dim]
        
        # Force direction vectors
        force_directions = token_i - token_j  # [batch, seq, seq, dim]
        force_magnitudes = torch.norm(force_directions, dim=-1, keepdim=True)
        normalized_forces = force_directions / (force_magnitudes + 1e-8)
        
        # Direction-sensitive force effects
        direction_scores = torch.einsum('bstn,hd->bshtn', normalized_forces, self.direction_modulator)
        force_field = direction_scores * torch.exp(-force_magnitudes)
        force_attention = torch.sum(force_field, dim=-1)  # [batch, heads, seq, seq]
        
        # Calculate topological edges
        xi = x.unsqueeze(2).expand(-1, -1, seq_len, -1)
        xj = x.unsqueeze(1).expand(-1, seq_len, -1, -1)
        pairs = torch.cat([xi, xj], dim=-1)
        
        # Basic connectivity
        edge_logits = self.edge_projector(pairs).permute(0, 3, 1, 2)  # [batch, heads, seq, seq]
        direct_edges = torch.sigmoid(edge_logits)
        
        # Multi-hop paths - calculate powers of adjacency matrix
        topo_paths = [direct_edges]
        current_paths = direct_edges
        
        for i in range(1, self.hop_levels):
            # Compute next-hop connections
            next_hop = torch.matmul(current_paths, direct_edges) / (seq_len ** 0.5)
            topo_paths.append(next_hop)
            current_paths = next_hop
        
        # Combine different hop lengths with learned weights
        topo_attention = sum(w * path for w, path in zip(F.softmax(self.hop_weights, dim=0), topo_paths))
        
        # Integrate force and topological attention
        balance = torch.sigmoid(self.force_topo_balance)
        combined_attention = balance * force_attention + (1 - balance) * topo_attention
        
        # Apply mask if provided
        if mask is not None:
            combined_attention = combined_attention + mask
            
        # Get attention weights and compute output
        weights = F.softmax(combined_attention, dim=-1)
        output = torch.matmul(weights, v)
        output = output.transpose(1, 2).contiguous().reshape(batch, seq_len, -1)
        
        return self.output(output)
```
Neural-inspired architecture that models the biological concept of myelin sheaths and nodes of Ranvier, enabling targeted computation and dynamic layer traversal based on content importance. Features reinforcement learning-based policy for optimized layer skipping.
 
 3. Reinforcement Learning Enhanced Attention
```python
        class Refiner:
            def __init__(self, states, actions, alpha=0.1, gamma=0.9, epsilon=0.1):
                self.states = states
                self.actions = actions
                self.R = {}
                 Q-learning for optimizing attention parameters
```

Integration of Q-learning to dynamically refine attention parameters, allowing the model to learn optimal attention spans through exploration and exploitation during training.
 
 4. Integrated Local-Global Attention
 
```python
    
        class IntegratedAttention(nn.Module):
            """Combines local adaptive span and global content-dependent attention with RL-based adaptation."""
            def __init__(self, dims, head, max_dist=512, win_size=256, max_span=384, temp_scale=0.01):
                 Hybrid attention combining multiple mechanisms

```
Combines sliding window attention with adaptive spans and global context awareness, creating a hybrid approach that balances efficiency and modeling capacity.
 
 Content-Dependent Update Mechanisms
 
```python

    def should_update_key(self, x: torch.Tensor) -> torch.Tensor:
        """Predict whether the key should be updated based on content."""
        avg_rep = x.mean(dim=1)
        return self.key_update_predictor(avg_rep) > self.update_threshold
```
 Implements neural predictors that determine whether keys and values should be updated based on content.
 
 Dynamic Layer Skipping
 
```python
    
    def decide_jump(self, policy, jump_weights, i, layerA, x, original_x, working_memory):
        """Decide whether to jump layers based on the policy network."""
        jump_prob = policy[:, 1] if i < layerA - 1 else torch.zeros_like(policy[:, 1])
        should_jump = (torch.rand_like(jump_prob) < jump_prob).any()
         Layer skipping logic

```
 Learns when to skip unnecessary computation through neural policy networks.
 
 Multi-Scale Processing
 
```python
    def slide_win(self, x, win_size, span_len, span_scale, mask=None):
        """Process input with sliding window attention."""
        batch, ctx, dims = x.size()
        num_windows = (ctx + win_size - 1) // win_size
         Sliding window implementation

```
 Rotary Embeddings
  
 - Implements quaternion mathematics for 3D rotations, enhancing positional encoding in transformer models.
 - Projects high-dimensional embeddings into 3D space for rotation and back to the original dimensionality, preserving geometric relationships.
 - Supports learnable rotation parameters, allowing the model to adapt positional embeddings dynamically.
 - Provides lightweight options for rotational embeddings, reducing computational overhead while maintaining flexibility.
```python
        def q_rotation(self, x, theta, u, v=None):
            eps = 1e-8
            u_norm = torch.norm(u, p=2)
            u = u / (u_norm + eps)
            w = torch.cos(theta / 2)
            vec = torch.sin(theta / 2) * u
            x_shape = x.shape
            x = x.reshape(-1, 3)
            uv_cross = torch.cross(u.unsqueeze(0), x)
            uuv_cross = torch.cross(u.unsqueeze(0), uv_cross)
            x_rot = x + 2 * (w * uv_cross + uuv_cross)
            return x_rot.reshape(*x_shape)
        

```

