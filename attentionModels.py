# Create attention head Q, K, V models for a single head
import torch

from attentionCalcs import rope_encode

class AttentionHead(torch.nn.Module):
    def __init__(self, embed_dim, head_dim, position_encoder_func = None):
        super(AttentionHead, self).__init__()
        self.query = torch.nn.Linear(embed_dim, head_dim)
        self.key = torch.nn.Linear(embed_dim, head_dim)
        self.value = torch.nn.Linear(embed_dim, head_dim)
        self.head_dim = head_dim
        self.position_encoder_func = position_encoder_func

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        if self.position_encoder_func is not None:
            Q = self.position_encoder_func(Q)
            K = self.position_encoder_func(K)

        # Perform scaled dot-product attention
        scale = K.shape[-1] ** 0.5
        scores = Q @ K.transpose(-2, -1)
        scaled_scores = scores / scale
        attention_weights = torch.softmax(scaled_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)

        return attention_output, Q, K, V, attention_weights

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, position_encoder_func=None):
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.head_dim = embed_dim // num_heads

        self.heads = torch.nn.ModuleList([
            AttentionHead(
                embed_dim,
                self.head_dim,
                position_encoder_func
            )
            for _ in range(num_heads)
        ])

        self.linear = torch.nn.Linear(
            self.head_dim * num_heads,
            embed_dim
        )

    def forward(self, x):
        head_results = [head(x) for head in self.heads]

        outputs = [r[0] for r in head_results]
        Qs = [r[1] for r in head_results]
        Ks = [r[2] for r in head_results]
        Vs = [r[3] for r in head_results]
        attention_weights = [r[4] for r in head_results]

        attention_output = torch.cat(outputs, dim=-1)

        return (
            self.linear(attention_output),
            attention_weights,
            Qs,
            Ks,
            Vs
        )

class OutputHead(torch.nn.Module):
    def __init__(self, embed_dim, output_dim):
        super(OutputHead, self).__init__()
        self.linear = torch.nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

class Embedder(torch.nn.Module):
    def __init__(self, input_dim, embed_dim):
        super(Embedder, self).__init__()
        self.embedder = torch.nn.Embedding(input_dim, embed_dim)

    def forward(self, x):
        return self.embedder(x)

class ToyTransformer(torch.nn.Module):
    def __init__(self, embedder, layer1, layer2, output_head, input_position_encoder=None):
        super().__init__()

        self.embedder = embedder
        self.layer1 = layer1
        self.layer2 = layer2
        self.output_head = output_head
        self.input_position_encoder = input_position_encoder

    def forward(self, x):
        
        x = self.embedder(x)
        if self.input_position_encoder is not None:
            x = self.input_position_encoder(x)

        residual_0 = x
        attn1, attn_weights1, Qs1, Ks1, Vs1 = self.layer1(x)
        x = x + attn1
        residual_1 = x
        attn2, attn_weights2, Qs2, Ks2, Vs2 = self.layer2(x)
        x = x + attn2
        residual_2 = x

        output = self.output_head(x)

        return (
            output,
            (attn_weights1, attn_weights2),
            (Qs1, Qs2),
            (Ks1, Ks2),
            (Vs1, Vs2),
            (residual_0, residual_1, residual_2)
        )