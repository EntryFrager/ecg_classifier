import torch
import torch.nn as nn

import hydra
from typing import Optional, List, Type


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        drop_prob: float = 0.0,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()

        self.conv_1 = nn.Conv1d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.batch_norm_1 = nn.BatchNorm1d(planes)
        self.relu_1 = nn.ReLU()
        self.conv_2 = nn.Conv1d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.batch_norm_2 = nn.BatchNorm1d(planes)
        self.relu_2 = nn.ReLU()

        self.drop_1 = nn.Dropout1d(p=drop_prob)
        self.drop_2 = nn.Dropout1d(p=drop_prob)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv_1(x)
        out = self.batch_norm_1(out)
        out = self.relu_1(out)
        out = self.drop_1(out)

        out = self.conv_2(out)
        out = self.batch_norm_2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu_2(out)
        out = self.drop_2(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        drop_prob: float = 0.0,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()

        self.conv_1 = nn.Conv1d(
            inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.batch_norm_1 = nn.BatchNorm1d(planes)
        self.drop_1 = nn.Dropout1d(p=drop_prob)
        self.relu_1 = nn.ReLU()
        self.conv_2 = nn.Conv1d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.batch_norm_2 = nn.BatchNorm1d(planes)
        self.drop_2 = nn.Dropout1d(p=drop_prob)
        self.relu_2 = nn.ReLU()
        self.conv_3 = nn.Conv1d(
            planes,
            planes * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.batch_norm_3 = nn.BatchNorm1d(planes * self.expansion)
        self.drop_3 = nn.Dropout1d(p=drop_prob)
        self.relu_3 = nn.ReLU()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv_1(x)
        out = self.batch_norm_1(out)
        out = self.relu_1(out)
        out = self.drop_1(out)

        out = self.conv_2(out)
        out = self.batch_norm_2(out)
        out = self.relu_2(out)
        out = self.drop_2(out)

        out = self.conv_3(out)
        out = self.batch_norm_3(out)
        out = self.relu_3(out)
        out = self.drop_3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: str,
        layers: List[int],
        num_classes: int = 1000,
        drop_prob_head: float = 0.5,
        drop_prob_backbone: float = 0.0,
    ):
        super().__init__()

        if isinstance(block, str):
            block = hydra.utils.get_class(block)

        self.inplanes = 64
        self.drop_prob_backbone = drop_prob_backbone

        self.conv_1 = nn.Conv1d(
            12, self.inplanes, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.batch_norm_1 = nn.BatchNorm1d(self.inplanes)
        self.relu = nn.ReLU()
        self.maxpool_1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer_1 = self._make_layer(block, 64, layers[0])
        self.layer_2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer_3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer_4 = self._make_layer(block, 512, layers[3], stride=2)

        self.drop = nn.Dropout1d(p=drop_prob_head)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_1(x)
        out = self.batch_norm_1(out)
        out = self.relu(out)
        out = self.maxpool_1(out)

        out = self.layer_1(out)
        out = self.layer_2(out)
        out = self.layer_3(out)
        out = self.layer_4(out)

        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.drop(out)
        out = self.fc(out)

        return out

    def _make_layer(
        self, block: Type[nn.Module], planes: int, blocks: int, stride: int = 1
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, self.drop_prob_backbone, downsample)
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, drop_prob=self.drop_prob_backbone)
            )

        return nn.Sequential(*layers)


class ResNetMeta(nn.Module):
    def __init__(
        self,
        block: str,
        layers: List[int],
        in_channels: int = 12,
        num_classes: int = 1000,
        drop_prob_head: float = 0.5,
        drop_prob_backbone: float = 0.0,
        in_channels_meta: int = 0,
        drop_prob_meta: float = 0.0,
    ):
        super().__init__()

        if isinstance(block, str):
            block = hydra.utils.get_class(block)

        self.inplanes = 64
        self.drop_prob_backbone = drop_prob_backbone

        self.backbone = nn.Sequential(
            nn.Conv1d(
                in_channels,
                self.inplanes,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(self.inplanes),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            self._make_layer(block, 64, layers[0]),
            self._make_layer(block, 128, layers[1], stride=2),
            self._make_layer(block, 256, layers[2], stride=2),
            self._make_layer(block, 512, layers[3], stride=2),
            nn.AdaptiveAvgPool1d(1),
        )

        self.meta_branch = nn.Sequential(
            nn.Linear(in_channels_meta, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=drop_prob_meta),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=drop_prob_meta),
        )

        self.head = nn.Sequential(
            nn.Linear(512 * block.expansion + 128, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=drop_prob_head),
            nn.Linear(512, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x_ecg: torch.Tensor, x_meta: torch.Tensor) -> torch.Tensor:
        out = self.backbone(x_ecg)
        out = torch.flatten(out, 1)

        out_meta = self.meta_branch(x_meta)
        out = torch.cat([out, out_meta], dim=1)

        out = self.head(out)

        return out

    def _make_layer(
        self, block: Type[nn.Module], planes: int, blocks: int, stride: int = 1
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, self.drop_prob_backbone, downsample)
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, drop_prob=self.drop_prob_backbone)
            )

        return nn.Sequential(*layers)
