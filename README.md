
#### Neural Network Optimizations and Embeddings


This repository contains custom implementations of frequency-adaptive optimization algorithms n-dimensional rotary positional embeddings and attentions for transformers and tranformer-like architectures that are nlp/asr focused. And a few other things. These naturally lend themselves to vision and multimodal. Everything here is in a forever wip state and tends to be buggy. 

I stress tranformer-like. 

Layer variance, (just like mother nature intended).

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
class FrequencyHandler:
    def analyze(self, grad_sample, n_bands, eps=1e-8):
        """Frequency analysis implementation using FFT"""
        freq_repr = torch.fft.rfft(grad_sample.float())
        freq_power = torch.abs(freq_repr)
         Normalize and divide into frequency bands


 ```
 
 1. Gradient Frequency Analysis
 The FAM optimizer analyzes gradient frequency spectra to dynamically adjust optimization parameters. This addresses the challenge that different parameter types (attention, embeddings, convolutions) require different update strategies.
 
 2. Parameter-Specific Handlers
```python
        class AttentionFrequencyHandler(FrequencyHandler):
            """Specialized handler for attention layers"""
            def analyze(self, grad_sample, n_bands, eps=1e-8):
                 Attention layers often have important high-frequency patterns
                 Use more bands in high frequencies
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
 
 4. Debug and Monitoring Tools
 
 Includes debug tools to track frequency band distribution across training, helping identify optimization challenges. (this is mostly for my sanity)
 
 n-Dimensional Rotary Embedding (Givens-Quaternion with regular RoPE fall back)
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

