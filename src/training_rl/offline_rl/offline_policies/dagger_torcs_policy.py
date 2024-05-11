from typing import Dict, Any

import torch
from tianshou.policy import ImitationPolicy
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


dagger_config = {
    "input_dim": 19,  # number of lidar rays
    "output_dim": 1,  # steer direction
}


def dagger_torcs_default_config() -> Dict[str, Any]:
    return dagger_config


class DaggerTorcsPolicy(nn.Module):
    """
    Simple policy to use in our torcs environment where the input corresponds to the 19 lidar rays and the output
    corresponds to the steer value
    """
    def __init__(self, input_dim=19, output_dim=1):
        super(DaggerTorcsPolicy, self).__init__()
        self.shared_layer = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.angle_output = nn.Linear(512, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.shared_layer(x)
        x = self.angle_output(x)
        x = self.tanh(x)
        return x


def create_dagger_torcs_policy_from_dict(
    policy_config: Dict[str, Any] = None,
):
    if policy_config is None:
        policy_config = dagger_config
    return DaggerTorcsPolicy(input_dim=policy_config["input_dim"], output_dim=policy_config["output_dim"])


# ToDo: This should be implemented in Tianshou. But as DAGGER involved an aggregation of the buffer this is a faster
#   way for now.
def model_dagger_fit(
    input_data: torch.Tensor,
    target_data: torch.Tensor,
    model: nn.Module | ImitationPolicy,
    batch_size=128,
    epochs=1,
    shuffle=True,
):
    target_data = target_data.reshape(-1, 1)
    dataset = TensorDataset(input_data, target_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    criterion = torch.nn.MSELoss()

    torch_model = model if not isinstance(model, ImitationPolicy) else model.actor

    optimizer = torch.optim.Adam(torch_model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs_batch, targets_batch in dataloader:
            optimizer.zero_grad()
            outputs = torch_model(inputs_batch)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = criterion(outputs, targets_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(dataloader):.4f}')
