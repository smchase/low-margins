# export_mlp_to_onnx.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx

class MLP(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((8, 8))
        self.fc = nn.Linear(1 * 8 * 8, num_classes)

    def forward(self, x):
        # For export we’ll pass 4D input, so this branch won’t run.
        if x.dim() == 2:
            x = x.view(x.size(0), 1, 28, 28)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def main():
    model = MLP(num_classes=10)
    # If you have trained weights, load them here:
    model.load_state_dict(torch.load("model_weights.pth", map_location="cpu"))
    model.eval()

    dummy = torch.randn(1, 1, 28, 28)  # (N,C,H,W)

    torch.onnx.export(
        model,
        dummy,
        "mlp.onnx",
        input_names=["input"],
        output_names=["logits"],
        opset_version=18,
        # dynamic_shapes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )
    print("✅ Exported to mlp.onnx")

    imported_model = onnx.load("mlp.onnx", load_external_data=True)
    onnx.save(imported_model, "mlp_single.onnx")

    print("✅ Imported model saved to mlp_single.onnx")


if __name__ == "__main__":
    main()
