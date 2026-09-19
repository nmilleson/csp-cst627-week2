# Generate 8-token synthetic sorting dataset for attention model
import torch

def generate_synthetic_sorting_dataset(num_samples, seq_length):
    # Generate random integers between 1 and sequence length (inclusive)
    data = torch.randint(0, seq_length, (num_samples, seq_length))
    sorted_data, _ = torch.sort(data, dim=1)
    return data, sorted_data

# Return split dataset with train/test/validate
def split_dataset(data, sorted_data, train_ratio=0.8, test_ratio=0.1):
    num_samples = data.shape[0]
    train_end = int(num_samples * train_ratio)
    test_end = train_end + int(num_samples * test_ratio)

    train_data = data[:train_end]
    train_sorted = sorted_data[:train_end]

    test_data = data[train_end:test_end]
    test_sorted = sorted_data[train_end:test_end]

    validate_data = data[test_end:]
    validate_sorted = sorted_data[test_end:]

    return (train_data, train_sorted), (test_data, test_sorted), (validate_data, validate_sorted)