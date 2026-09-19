# Generate dataset
import torch
from torch import nn
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from attentionCalcs import rope_encode, sinusoidal_encode
from datasetGen import generate_synthetic_sorting_dataset, split_dataset


num_samples = 10000
seq_length = 8
data, sorted_data = generate_synthetic_sorting_dataset(num_samples, seq_length)
train_set, test_set, validate_set = split_dataset(data, sorted_data)

# Create 2-layer, 2-head multi-head attention model
from attentionModels import MultiHeadAttention, OutputHead, Embedder, ToyTransformer

embed_dim = 8
num_heads = 2
output_dim = 8

embedder_rope = Embedder(input_dim=seq_length, embed_dim=embed_dim)
layer1_rope = MultiHeadAttention(embed_dim, num_heads, position_encoder_func=rope_encode)
layer2_rope = MultiHeadAttention(embed_dim, num_heads, position_encoder_func=rope_encode)
output_head_rope = OutputHead(embed_dim, output_dim)
rope_transformer = ToyTransformer(embedder_rope, layer1_rope, layer2_rope, output_head_rope, input_position_encoder=None)

embedder = Embedder(input_dim=seq_length, embed_dim=embed_dim)
layer1 = MultiHeadAttention(embed_dim, num_heads, position_encoder_func=None)
layer2 = MultiHeadAttention(embed_dim, num_heads, position_encoder_func=None)
output_head = OutputHead(embed_dim, output_dim)
sinusoidal_transformer = ToyTransformer(embedder, layer1, layer2, output_head, input_position_encoder=sinusoidal_encode)

def evaluate(model, data_loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    total_correct_sequences = 0
    total_sequences = 0

    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            target = target.to(device)

            output, _, _, _, _, _ = model(data)

            loss = criterion(
                output.reshape(-1, output.size(-1)),
                target.reshape(-1)
            )

            total_loss += loss.item()

            preds = output.argmax(dim=-1)

            # Token accuracy
            total_correct += (preds == target).sum().item()
            total_tokens += target.numel()

            # Whole-sequence accuracy
            correct_sequences = (preds == target).all(dim=1)
            total_correct_sequences += correct_sequences.sum().item()
            total_sequences += target.size(0)

    avg_loss = total_loss / len(data_loader)
    token_accuracy = total_correct / total_tokens
    sequence_accuracy = total_correct_sequences / total_sequences

    return avg_loss, token_accuracy, sequence_accuracy

# Define the RoPE training loop
def train(model, train_loader, validate_loader, criterion, optimizer, scheduler=None, num_epochs=10):
    
    for epoch in range(num_epochs):
        total_train_loss = 0.0
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output, _, _, _, _, _= model(data)
            loss = criterion(
                output.reshape(-1, output.size(-1)),
                target.reshape(-1)
            )
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        train_loss = total_train_loss / len(train_loader)

        val_loss, val_token_accuracy, val_sequence_accuracy = evaluate(
            model,
            validate_loader,
            criterion,
            device
        )
        if scheduler is not None:
            scheduler.step(val_loss)

        print(
            f"Epoch {epoch+1}: "
            f"train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, "
            f"val_token_acc={val_token_accuracy:.4f}, "
            f"val_sequence_acc={val_sequence_accuracy:.4f}"
        )


# Define the device for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move model components to the device
embedder_rope.to(device)
embedder.to(device)
layer1_rope.to(device)
layer2_rope.to(device)
layer1.to(device)
layer2.to(device)
output_head_rope.to(device)
output_head.to(device)

# Move training data to the device
train_data, train_targets = train_set

train_dataset = TensorDataset(train_data, train_targets)
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

validate_data, validate_targets = validate_set

validate_dataset = TensorDataset(validate_data, validate_targets)
validate_loader = DataLoader(
    validate_dataset,
    batch_size=32,
    shuffle=False
)

# Move training data to the device
train_data = train_data.to(device)
train_targets = train_targets.to(device)

criterion = torch.nn.CrossEntropyLoss()

rope_optimizer = optim.Adam(rope_transformer.parameters())
rope_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    rope_optimizer,
    mode="min",
    factor=0.5,
    patience=10
)

sinusoidal_optimizer = optim.Adam(sinusoidal_transformer.parameters())
sinusoidal_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    sinusoidal_optimizer,
    mode="min",
    factor=0.5,
    patience=10
)


# Training loops
train(model=rope_transformer, train_loader=train_loader, validate_loader=validate_loader, criterion=criterion, optimizer=rope_optimizer, scheduler=rope_scheduler, num_epochs=150)
train(model=sinusoidal_transformer, train_loader=train_loader, validate_loader=validate_loader, criterion=criterion, optimizer=sinusoidal_optimizer, scheduler=sinusoidal_scheduler, num_epochs=150)

# Save model components
torch.save(rope_transformer.state_dict(), "rope_transformer.pth")
torch.save(sinusoidal_transformer.state_dict(), "sinusoidal_transformer.pth")