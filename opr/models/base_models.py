"""Interfaces and meta-models definitions."""
from typing import Dict, Optional, Union

import MinkowskiEngine as ME  # noqa: N817

import torch
from torch import Tensor, nn


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        x += identity
        x = self.relu(x)
        return x


class ResNet18(nn.Module):
    layers = (64, 64, 128, 256)
    def __init__(self, in_channels=64, out_channels=512):
        super(ResNet18, self).__init__()
        self.in_channels = in_channels
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # resnet layers
        self.layer1 = self.__make_layer(64, 64, stride=1)
        self.layer2 = self.__make_layer(64, 128, stride=2)
        self.layer3 = self.__make_layer(128, 256, stride=2)
        self.layer4 = self.__make_layer(256, out_channels, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def __make_layer(self, in_channels, out_channels, stride):
        identity_downsample = None
        if stride != 1:
            identity_downsample = self.identity_downsample(in_channels, out_channels)

        return nn.Sequential(
            ResNetBlock(in_channels, out_channels, identity_downsample=identity_downsample, stride=stride),
            ResNetBlock(out_channels, out_channels)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        return x

    def identity_downsample(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels)
        )


class ResNet10Block(nn.Module):
    """"""
    def __init__(self, in_channels, out_channels):
        super(ResNet10Block, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.res_block1 = self.__make_layer(in_channels=in_channels, out_channels=in_channels, stride=1)
        self.res_block2 = self.__make_layer(in_channels=in_channels, out_channels=in_channels, stride=1)
        self.res_block3 = self.__make_layer(in_channels=in_channels, out_channels=in_channels, stride=1)
        self.res_block4 = self.__make_layer(in_channels=in_channels, out_channels=in_channels, stride=1)
        self.res_block5 = self.__make_layer(in_channels=in_channels, out_channels=out_channels, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def __make_layer(self, in_channels, out_channels, stride):
        identity_downsample = None
        if stride != 1:
            identity_downsample = self.identity_downsample(in_channels, out_channels)

        return nn.Sequential(
            ResNetBlock(in_channels, out_channels, identity_downsample=identity_downsample, stride=stride),
            ResNetBlock(out_channels, out_channels)
        )

    def forward(self, x):
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        x = self.res_block5(x)

        x = self.avgpool(x)
        return x


class SemanticEmbeddingV1(nn.Module):
    # TODO: Попробовать сделать на основе резнет выше
    # так просто модели заменить не?
    def __init__(self, in_channels=65, out_channels=512):
        super(SemanticEmbeddingV1, self).__init__()

        self.resnet18 = ResNet18(in_channels=in_channels, out_channels=512)
        # self.resnet18 = ResNet18FPNExtractor(pretrained=False)
        # resnet18layers = list(self.resnet18.layers)
        # resnet18layers = list(self.resnet18.layers)
        # resnet18layers[0] = in_channels
        # self.resnet18.layers = tuple(resnet18layers)
        self.final_conv = nn.Conv2d(in_channels=512, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        x = self.resnet18(x)
        embedding = self.final_conv(x)
        return embedding


class FusionSemanticV1(nn.Module):
    def __init__(self, in_channels=65, out_channels=128):
        super(FusionSemanticV1, self).__init__()
        self.sem_module1 = SemanticEmbeddingV1(in_channels=in_channels, out_channels=out_channels)
        self.sem_module2 = SemanticEmbeddingV1(in_channels=in_channels, out_channels=out_channels)

        self.flatten = nn.Flatten()

        self.semantic_fusion = nn.Sequential(
            nn.Linear(in_features=128 * 2, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=128)
        )

    def forward(self, x1, x2):
        embedding1 = self.flatten(self.sem_module1(x1))
        embedding2 = self.flatten(self.sem_module2(x2))

        embedding = torch.concat((embedding1, embedding2), axis=1)
        embedding = self.semantic_fusion(embedding)
        return embedding


class SemanticEmbeddingV2(nn.Module):
    """Для одного тензора карт активации классов сегментации и для одного изображения, соответсвующего маске"""
    def __init__(self, in_channels_image=3, in_channels_mask=65, out_channels=32):
        super(SemanticEmbeddingV2, self).__init__()

        self.main_branch_block1 = nn.Sequential(
            ResNet10Block(in_channels=in_channels_mask, out_channels=64),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.main_branch_block2 = nn.Sequential(
            ResNet10Block(in_channels=64*2, out_channels=128),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.main_branch_block3 = nn.Sequential(
            ResNet10Block(in_channels=128 * 2, out_channels=256),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.sub_branch_block1 = nn.Sequential(
            ResNet10Block(in_channels=in_channels_image, out_channels=64),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.sub_branch_block2 = nn.Sequential(
            ResNet10Block(in_channels=64 * 2, out_channels=128),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.sub_branch_block3 = nn.Sequential(
            ResNet10Block(in_channels=128 * 2, out_channels=256),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.final_conv = nn.Conv2d(in_channels=256*2, out_channels=out_channels, kernel_size=3, padding=1)

    def forward(self, mask, x1):
        features = self.main_branch_block1(mask)
        sub_features = self.sub_branch_block1(x1)
        features = torch.concat((features, sub_features), axis=1)

        features = self.main_branch_block2(features)
        sub_features = self.sub_branch_block2(sub_features)
        features = torch.concat((features, sub_features), axis=1)

        features = self.main_branch_block3(features)
        sub_features = self.sub_branch_block3(sub_features)
        features = torch.concat((features, sub_features), axis=1)

        embedding = self.final_conv(features)
        return embedding


class FusionSemanticV2(nn.Module):
    def __init__(self, in_channels_image=3, in_channels_mask=65, out_channels=128):
        super(FusionSemanticV2, self).__init__()
        self.sem_module1 = SemanticEmbeddingV2(in_channels_mask=in_channels_image, out_channels=out_channels)
        self.sem_module2 = SemanticEmbeddingV2(in_channels_mask=in_channels_image, out_channels=out_channels)

        self.flatten = nn.Flatten()

        # TODO: надо посчитать in_features
        self.semantic_fusion = nn.Sequential(
            nn.Linear(in_features=128 * 2, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=128)
        )

    def forward(self, mask1, x1, mask2, x2):
        embedding1 = self.flatten(self.sem_module1(mask1, x1))
        embedding2 = self.flatten(self.sem_module2(mask2, x2))

        # TODO: помнить про длины векторов
        embedding = torch.concat((embedding1, embedding2), axis=1)
        embedding = self.semantic_fusion(embedding)
        return embedding


class ImageFeatureExtractor(nn.Module):
    """Interface class for image feature extractor module."""

    def __init__(self):
        """Interface class for image feature extractor module."""
        super().__init__()

    def forward(self, image: Tensor) -> Tensor:  # noqa: D102
        raise NotImplementedError()


class ImageHead(nn.Module):
    """Interface class for image head module."""

    def __init__(self):
        """Interface class for image head module."""
        super().__init__()

    def forward(self, feature_map: Tensor) -> Tensor:  # noqa: D102
        raise NotImplementedError()


class ImageModule(nn.Module):
    """Meta-module for image branch. Combines feature extraction backbone and head modules."""

    def __init__(
        self,
        backbone: ImageFeatureExtractor,
        head: ImageHead,
    ):
        """Meta-module for image branch.

        Args:
            backbone (ImageFeatureExtractor): Image feature extraction backbone.
            head (ImageHead): Image head module.
        """
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
        x = self.backbone(x)
        x = self.head(x)
        return x


class CloudFeatureExtractor(nn.Module):
    """Interface class for cloud feature extractor module."""

    sparse: bool

    def __init__(self):
        """Interface class for cloud feature extractor module."""
        super().__init__()
        assert self.sparse is not None

    def forward(self, cloud: Union[Tensor, ME.SparseTensor]) -> Union[Tensor, ME.SparseTensor]:  # noqa: D102
        raise NotImplementedError()


class CloudHead(nn.Module):
    """Interface class for cloud head module."""

    sparse: bool

    def __init__(self):
        """Interface class for cloud head module."""
        super().__init__()
        assert self.sparse is not None

    def forward(self, feature_map: Union[Tensor, ME.SparseTensor]) -> Tensor:  # noqa: D102
        raise NotImplementedError()


class CloudModule(nn.Module):
    """Meta-module for cloud branch. Combines feature extraction backbone and head modules."""

    def __init__(
        self,
        backbone: CloudFeatureExtractor,
        head: CloudHead,
    ):
        """Meta-module for cloud branch.

        Args:
            backbone (CloudFeatureExtractor): Cloud feature extraction backbone.
            head (CloudHead): Cloud head module.

        Raises:
            ValueError: If incompatible cloud backbone and head are given.
        """
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.sparse = self.backbone.sparse
        if self.backbone.sparse != self.head.sparse:
            raise ValueError("Incompatible cloud backbone and head")

    def forward(self, x: Union[Tensor, ME.SparseTensor]) -> Tensor:  # noqa: D102
        if self.sparse:
            assert isinstance(x, ME.SparseTensor)
        else:
            assert isinstance(x, Tensor)
        x = self.backbone(x)
        x = self.head(x)
        return x


class FusionModule(nn.Module):
    """Interface class for fusion module."""

    def __init__(self):
        """Interface class for fusion module."""
        super().__init__()

    def forward(self, data: Dict[str, Union[Tensor, ME.SparseTensor]]) -> Tensor:  # noqa: D102
        raise NotImplementedError()


class ComposedModel(nn.Module):
    """Composition model for multimodal architectures."""

    sparse_cloud: Optional[bool] = None

    def __init__(
        self,
        image_module: Optional[ImageModule] = None,
        cloud_module: Optional[CloudModule] = None,
        semantic_module: Optional[FusionSemanticV1] = None,
        fusion_module: Optional[FusionModule] = None,
    ) -> None:
        """Composition model for multimodal architectures.

        Args:
            image_module (ImageModule, optional): Image modality branch. Defaults to None.
            cloud_module (CloudModule, optional): Cloud modality branch. Defaults to None.
            semantic_module: semantic modality branch
            fusion_module (FusionModule, optional): Module to fuse different modalities. Defaults to None.
        """
        super().__init__()

        self.image_module = image_module
        self.cloud_module = cloud_module
        self.semantic_module = semantic_module
        self.fusion_module = fusion_module
        if self.cloud_module:
            self.sparse_cloud = self.cloud_module.sparse

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Optional[Tensor]]:  # noqa: D102
        out_dict: Dict[str, Optional[Tensor]] = {
            "image": None,
            "cloud": None,
            "semantic": None,
            "fusion": None,
        }

        if self.image_module is not None:
            out_dict["image"] = self.image_module(batch["images"])

        if self.cloud_module is not None:
            if self.sparse_cloud:
                cloud = ME.SparseTensor(features=batch["features"], coordinates=batch["coordinates"])
            else:
                raise NotImplementedError("Currently we support only sparse cloud modules.")
            out_dict["cloud"] = self.cloud_module(cloud)

        if self.semantic_module is not None:
            # TODO: посмотреть, что пихать в семантик модуль из даталоадера
            out_dict["semantic"] = self.semantic_module()

        if self.fusion_module is not None:
            out_dict["fusion"] = self.fusion_module(out_dict)

        return out_dict
