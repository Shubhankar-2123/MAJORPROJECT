# # train_static_model.py

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import DataLoader, TensorDataset
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report


# # -----------------------------
# # Define model
# # -----------------------------
# class StaticModel(nn.Module):
#     def __init__(self, input_dim, num_classes):
#         super().__init__()
#         self.fc1 = nn.Linear(input_dim, 256)
#         self.drop1 = nn.Dropout(0.3)
#         self.fc2 = nn.Linear(256, 128)
#         self.drop2 = nn.Dropout(0.3)
#         self.fc3 = nn.Linear(128, num_classes)
    
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = self.drop1(x)
#         x = F.relu(self.fc2(x))
#         x = self.drop2(x)
#         return self.fc3(x)

# if __name__ == "__main__":
#     # -----------------------------
#     # Load static keypoint data
#     # -----------------------------
#     X = np.load("data/processed/X_train_static.npy")  # path to your static keypoints
#     y = np.load("data/processed/y_train_static.npy")  # path to your labels

#     # Convert labels to strings and encode
#     y = np.array(y).astype(str)
#     le = LabelEncoder()
#     y = le.fit_transform(y)

#     # -----------------------------
#     # Train-test split
#     # -----------------------------
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42, stratify=y
#     )

#     # -----------------------------
#     # Normalize data safely
#     # -----------------------------
#     scaler = StandardScaler()
#     X_train[np.isnan(X_train)] = 0
#     X_train[np.isinf(X_train)] = 0
#     X_test[np.isnan(X_test)] = 0
#     X_test[np.isinf(X_test)] = 0

#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)

#     # Convert to torch tensors
#     X_train = torch.tensor(X_train, dtype=torch.float32)
#     y_train = torch.tensor(y_train, dtype=torch.long)
#     X_test  = torch.tensor(X_test, dtype=torch.float32)
#     y_test  = torch.tensor(y_test, dtype=torch.long)

#     # -----------------------------
#     # Create DataLoader
#     # -----------------------------
#     batch_size = 32
#     train_dataset = TensorDataset(X_train, y_train)
#     test_dataset  = TensorDataset(X_test, y_test)
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     test_loader  = DataLoader(test_dataset, batch_size=batch_size)


#     input_dim = X_train.shape[1]
#     num_classes = len(np.unique(y_train))
#     model = StaticModel(input_dim, num_classes)

#     # -----------------------------
#     # Training setup
#     # -----------------------------
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)

#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # small lr

#     # -----------------------------
#     # Training loop
#     # -----------------------------
#     num_epochs = 50

#     for epoch in range(1, num_epochs + 1):
#         model.train()
#         running_loss = 0.0

#         for xb, yb in train_loader:
#             xb, yb = xb.to(device), yb.to(device)
#             optimizer.zero_grad()
#             outputs = model(xb)
#             loss = criterion(outputs, yb)
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # gradient clipping
#             optimizer.step()
#             running_loss += loss.item() * xb.size(0)

#         epoch_loss = running_loss / len(train_loader.dataset)

#         # Validation accuracy
#         model.eval()
#         correct = 0
#         total = 0
#         with torch.no_grad():
#             for xb, yb in test_loader:
#                 xb, yb = xb.to(device), yb.to(device)
#                 outputs = model(xb)
#                 _, predicted = torch.max(outputs, 1)
#                 total += yb.size(0)
#                 correct += (predicted == yb).sum().item()
#         acc = correct / total

#         print(f"Epoch {epoch:02d}: Loss={epoch_loss:.4f}, Val_Acc={acc:.4f}")

#     # -----------------------------
#     # Evaluate final performance
#     # -----------------------------
#     model.eval()
#     y_pred = []
#     y_true = []

#     with torch.no_grad():
#         for xb, yb in test_loader:
#             xb = xb.to(device)
#             outputs = model(xb)
#             _, predicted = torch.max(outputs, 1)
#             y_pred.extend(predicted.cpu().numpy())
#             y_true.extend(yb.numpy())

#     print("\nClassification Report:\n")
#     print(classification_report(y_true, y_pred, digits=4))


#     # -----------------------------
#     # Quick sanity check on one sample
#     # -----------------------------
#     sample = X_test[0].unsqueeze(0).to(device)
#     with torch.no_grad():
#         output = model(sample)
#         pred = torch.argmax(output, dim=1).cpu().numpy()

#     print("\nSanity check:")
#     print("True label:", le.inverse_transform([y_test[0].cpu().numpy()])[0])
#     print("Predicted :", le.inverse_transform(pred)[0])



#     import joblib
#     import os

#     os.makedirs("models", exist_ok=True)

#     # Save model weights
#     torch.save(model.state_dict(), "models/static_model.pth")
#     print("✅ Model saved at models/static_model.pth")

#     # Save LabelEncoder
#     joblib.dump(le, "models/static_label_encoder.pkl")
#     print("✅ Label encoder saved at models/static_label_encoder.pkl")

#     # Save Scaler
#     joblib.dump(scaler, "models/static_scaler.pkl")
#     print("✅ Scaler saved at models/static_scaler.pkl")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

class StaticModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.drop1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)
        self.drop2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop2(x)
        return self.fc3(x)

def train_static_model():
    # Load processed splits and artifacts produced by preprocess_data
    X_train = np.load("data/processed/X_train_static.npy")
    X_test = np.load("data/processed/X_test_static.npy")
    y_train = np.load("data/processed/y_train_static.npy")
    y_test = np.load("data/processed/y_test_static.npy")

    # Load the exact LabelEncoder and StandardScaler that define the label mapping and feature scaling
    # This guarantees labels like 1-9 and A-Z instead of 0..N indices-as-strings
    le = joblib.load("data/processed/static_label_encoder.pkl")
    scaler = joblib.load("data/processed/static_scaler.pkl")

    # Convert to tensors (data is already scaled by preprocess step; do not re-scale here)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # DataLoader
    batch_size = 32
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    input_dim = X_train_tensor.shape[1]
    num_classes = len(getattr(le, 'classes_', np.unique(y_train)))

    model = StaticModel(input_dim, num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 50
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                outputs = model(xb)
                _, predicted = torch.max(outputs, 1)
                total += yb.size(0)
                correct += (predicted == yb).sum().item()
        acc = correct / total
        print(f"Epoch {epoch:02d}: Loss={epoch_loss:.4f}, Val_Acc={acc:.4f}")

    # Classification report
    model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            outputs = model(xb)
            _, predicted = torch.max(outputs, 1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(yb.numpy())

    print("\nClassification Report:\n")
    # Report with human-readable class names
    labels_in_use = sorted(set(y_true) | set(y_pred))
    target_names = le.inverse_transform(labels_in_use)
    print(classification_report(y_true, y_pred, labels=labels_in_use, target_names=target_names, digits=4))

    # Save model and copy preprocessing tools to models/ for inference
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/static_model.pth")
    joblib.dump(le, "models/static_label_encoder.pkl")
    joblib.dump(scaler, "models/static_scaler.pkl")
    print("Model, LabelEncoder and Scaler saved to models/.")

if __name__ == "__main__":
    train_static_model()

