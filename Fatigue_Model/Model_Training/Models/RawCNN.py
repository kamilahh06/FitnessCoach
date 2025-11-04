import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dilation=1, dropout=0.35):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=5, padding=2*dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=5, padding=2*dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(self.bn2(self.conv2(x)))
        return F.relu(x + residual)

class RawCNNRegressor(nn.Module):
    def __init__(self, input_channels=1, dropout=0.35):
        super().__init__()
        self.block1 = ResidualBlock(input_channels, 32, dilation=1, dropout=dropout)
        self.block2 = ResidualBlock(32, 64, dilation=2, dropout=dropout)
        self.block3 = ResidualBlock(64, 128, dilation=4, dropout=dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)



# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class ResidualBlock(nn.Module):
#     """1D residual block with dilation, BN, and ReLU."""
#     def __init__(self, in_ch, out_ch, dilation=1):
#         super().__init__()
#         self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=5, padding=2*dilation, dilation=dilation)
#         self.bn1 = nn.BatchNorm1d(out_ch)
#         self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=5, padding=2*dilation, dilation=dilation)
#         self.bn2 = nn.BatchNorm1d(out_ch)
#         self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

#     def forward(self, x):
#         residual = self.skip(x)
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = self.bn2(self.conv2(x))
#         return F.relu(x + residual)


# class RawCNNRegressor(nn.Module):
#     """Enhanced raw EMG CNN regressor with dilations, residuals, dropout, and weight decay support."""
#     def __init__(self, input_channels=1, dropout=0.3):
#         super().__init__()
#         self.block1 = ResidualBlock(input_channels, 32, dilation=1)
#         self.block2 = ResidualBlock(32, 64, dilation=2)
#         self.block3 = ResidualBlock(64, 128, dilation=4)
#         self.pool = nn.AdaptiveAvgPool1d(1)
#         self.dropout = nn.Dropout(dropout)
#         self.fc = nn.Linear(128, 1)
#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Conv1d):
#             nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
#         elif isinstance(m, nn.Linear):
#             nn.init.xavier_uniform_(m.weight)
#             nn.init.zeros_(m.bias)

#     def forward(self, x):
#         # Expecting x: [batch, seq_len] or [batch, 1, seq_len]
#         if x.ndim == 2:
#             x = x.unsqueeze(1)
#         x = self.block1(x)
#         x = self.block2(x)
#         x = self.block3(x)
#         x = self.pool(x).squeeze(-1)
#         x = self.dropout(x)
#         return self.fc(x)

# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F

# # class ResidualBlock(nn.Module):
# #     """1D residual block with dilation, batchnorm, dropout, and skip connection."""
# #     def __init__(self, in_ch, out_ch, dilation=1, dropout=0.2):
# #         super().__init__()
# #         self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=5, padding=2*dilation, dilation=dilation)
# #         self.bn1 = nn.BatchNorm1d(out_ch)
# #         self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=5, padding=2*dilation, dilation=dilation)
# #         self.bn2 = nn.BatchNorm1d(out_ch)
# #         self.dropout = nn.Dropout(dropout)
# #         self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

# #     def forward(self, x):
# #         residual = self.skip(x)
# #         x = F.relu(self.bn1(self.conv1(x)))
# #         x = self.dropout(self.bn2(self.conv2(x)))
# #         return F.relu(x + residual)


# # class RawCNNRegressor(nn.Module):
# #     """
# #     Enhanced CNN for EMG fatigue regression.
# #     Features:
# #       • Residual dilated conv blocks
# #       • Subject embedding support (optional)
# #       • Global pooling & dropout
# #       • Xavier + Kaiming initialization
# #     """
# #     def __init__(self, input_channels=1, num_subjects=None, dropout=0.35, embed_dim=8):
# #         super().__init__()

# #         # --- Optional subject embedding for personalization ---
# #         self.use_embedding = num_subjects is not None
# #         if self.use_embedding:
# #             self.subject_embed = nn.Embedding(num_subjects, embed_dim)
# #             fc_input_dim = 128 + embed_dim
# #         else:
# #             fc_input_dim = 128

# #         # --- Residual CNN backbone ---
# #         self.block1 = ResidualBlock(input_channels, 32, dilation=1, dropout=dropout)
# #         self.block2 = ResidualBlock(32, 64, dilation=2, dropout=dropout)
# #         self.block3 = ResidualBlock(64, 128, dilation=4, dropout=dropout)

# #         # --- Pooling + Fully Connected ---
# #         self.pool = nn.AdaptiveAvgPool1d(1)
# #         self.fc = nn.Sequential(
# #             nn.Dropout(dropout),
# #             nn.Linear(fc_input_dim, 64),
# #             nn.ReLU(),
# #             nn.Linear(64, 1)
# #         )

# #         self.apply(self._init_weights)

# #     def _init_weights(self, m):
# #         if isinstance(m, nn.Conv1d):
# #             nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
# #         elif isinstance(m, nn.Linear):
# #             nn.init.xavier_uniform_(m.weight)
# #             nn.init.zeros_(m.bias)

# #     def forward(self, x, subject_id=None):
# #         # x: [batch, seq_len] or [batch, 1, seq_len]
# #         if x.ndim == 2:
# #             x = x.unsqueeze(1)

# #         x = self.block1(x)
# #         x = self.block2(x)
# #         x = self.block3(x)

# #         x = self.pool(x).squeeze(-1)  # [batch, 128]

# #         # --- Concatenate subject embedding if available ---
# #         if self.use_embedding and subject_id is not None:
# #             emb = self.subject_embed(subject_id)
# #             x = torch.cat([x, emb], dim=1)

# #         return self.fc(x)