from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class ConvBlock(torch.nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int=0,
        stride: int=1,
        groups: int=1
    ):
        
        super().__init__()

        layers = [
            torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                groups=groups
            ),
            torch.nn.BatchNorm2d(
                out_channels
            ),
            torch.nn.ReLU6()
        ]

        self.block = torch.nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EncoderBlock(torch.nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        include_dropout: bool=False
    ):
        
        super().__init__()
        
        layers = [
            ConvBlock(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            ConvBlock(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
            )
        ]

        if include_dropout:
            layers.append(torch.nn.Dropout(p=.2))

        self.block = torch.nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class InvResEncoderBlock(torch.nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        block_reps: int,
        include_dropout: bool=False
    ):
        
        super().__init__()

        layers = list()

        current_input = in_channels
        for i in range(block_reps):

            stride = 1

            if i == 0:
                stride = 2
            
            layers.append(
                InvertedResBlock(
                    current_input,
                    out_channels,
                    exp_factor=4,
                    stride=stride
                )
            )

            current_input = out_channels
        
        self.block = torch.nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.block(x)


class UpsampleBlock(torch.nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        include_dropout: bool=False
    ):
        
        super().__init__()

        layers = [
            torch.nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                output_padding=1,
                stride=2
            ),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU6()
        ]

        if include_dropout:
            layers.append(
                torch.nn.Dropout(p=.2)
            )

        self.block = torch.nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return self.block(x)


class InvertedResBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        exp_factor: int,
        stride: int,
        **kwargs
    ) -> None:
        """
            Inverted residual block for mobile net v2

            Args:
                in_channels: int, number of input channels
                out_channels: int, number of output channels
                exp_factor: int, factor by which to expand the linear bottleneck
                stride: int, stride length for expansion layer
        """
        super().__init__()

        exp_channel_size = in_channels * exp_factor

        layers = [
            torch.nn.Conv2d(
                in_channels,
                out_channels=exp_channel_size,
                kernel_size=1
            ),
            torch.nn.BatchNorm2d(exp_channel_size),
            torch.nn.ReLU6(),
            torch.nn.Conv2d(
                exp_channel_size,
                out_channels=exp_channel_size,
                kernel_size=3,
                padding=1,
                stride=stride,
                groups=exp_channel_size # depthwise convolution
            ),
            torch.nn.BatchNorm2d(exp_channel_size),
            torch.nn.ReLU6(),
            torch.nn.Conv2d( # linear bottleneck
                exp_channel_size,
                out_channels=out_channels,
                kernel_size=1
            )
        ]

        # define the residual connection
        self.residual = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=stride
        )
        
        self.block = torch.nn.Sequential(*layers)
    
    def forward(self, x: torch.tensor) -> torch.tensor:

        return self.block(x) + self.residual(x)

class Classifier(nn.Module):

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 6,
    ):
        """
        A convolutional network for image classification.

        Args:
            in_channels: int, number of input channels
            num_classes: int
        """
        super().__init__()

        self.num_classes = num_classes
        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))

        block_arch_def = [
            {
                'exp_factor': 6,
                'out_channels': 32,
                'reps': 3,
                'stride': 2 
            },
            {
                'exp_factor': 6,
                'out_channels': 64,
                'reps': 4,
                'stride': 2 
            },
            {
                'exp_factor': 6,
                'out_channels': 96,
                'reps': 3,
                'stride': 1 
            },
            {
                'exp_factor': 6,
                'out_channels': 160,
                'reps': 3,
                'stride': 2 
            },
            {
                'exp_factor': 6,
                'out_channels': 320,
                'reps': 1,
                'stride': 1 
            },
        ]

        layers = [
            torch.nn.Conv2d(
                in_channels,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU6()
        ]

        o = 32
        for res_args in block_arch_def:

            for n in range(res_args['reps']):

                block_out_channels = res_args['out_channels']
                block_exp_factor = res_args['exp_factor']
                block_stride = res_args['stride']

                if n > 0:
                    block_stride = 1

                layers.append(
                    InvertedResBlock(
                        o,
                        out_channels=block_out_channels,
                        exp_factor=block_exp_factor,
                        stride=block_stride
                    )
                )

                o = res_args['out_channels']

        # final layers
        layers.append(torch.nn.Conv2d(o, 1280, kernel_size=1))
        layers.append(torch.nn.BatchNorm2d(1280))
        layers.append(torch.nn.ReLU6())
        layers.append(torch.nn.AdaptiveAvgPool2d(1))
        layers.append(torch.nn.ReLU6())
        layers.append(torch.nn.Dropout(p=.2))
        layers.append(torch.nn.Conv2d(1280, num_classes, kernel_size=1))
        
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, h, w) image

        Returns:
            tensor (b, num_classes) logits
        """
        # optional: normalizes the input
        # z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]


        # TODO: replace with actual forward pass
        logits = self.model(x).view(-1, self.num_classes)

        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Used for inference, returns class labels
        This is what the AccuracyMetric uses as input (this is what the grader will use!).
        You should not have to modify this function.

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            pred (torch.LongTensor): class labels {0, 1, ..., 5} with shape (b, h, w)
        """
        return self(x).argmax(dim=1)



class Detector(torch.nn.Module):

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 3,
    ):
        """
        A single model that performs segmentation and depth regression

        Args:
            in_channels: int, number of input channels
            num_classes: int
        """
        super().__init__()

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))

        # first conv layer
        self.first_layer = ConvBlock(
            in_channels,
            32,
            kernel_size=3,
            padding=1
        )

        # encoding layers
        self.encoder_layers = torch.nn.ModuleList()
        current_output_size = 32
        encoding_layers = 3

        for layer in range(encoding_layers):

            include_dropout = False

            if layer == (encoding_layers - 1):
                include_dropout = True

            self.encoder_layers.append(
                InvResEncoderBlock(
                    current_output_size,
                    current_output_size * 2,
                    block_reps=2,
                    include_dropout=include_dropout
                )
            )

            current_output_size *= 2

        
        # bottleneck layers
        self.bottleneck_layers = [
            ConvBlock(
                current_output_size,
                current_output_size * 2,
                kernel_size=3,
                padding=1
            ),
            torch.nn.Dropout(p=.5),
            ConvBlock(
                current_output_size * 2,
                current_output_size,
                kernel_size=3,
                padding=1
            )
        ]

        self.bottleneck = torch.nn.Sequential(*self.bottleneck_layers)
        
        # define decode layers
        self.decode_layers = torch.nn.ModuleList()
        for i, _ in enumerate(self.encoder_layers):

            include_dropout = False

            if i == (encoding_layers - 1):
                include_dropout = True

            first_decode_layer = ConvBlock(
                current_output_size,
                current_output_size,
                kernel_size=3,
                padding=1
            )

            upsample_layer = UpsampleBlock(
                current_output_size * 2,
                current_output_size // 2,
                include_dropout
            )

            decode_package = torch.nn.ModuleList([first_decode_layer, upsample_layer])
            self.decode_layers.append(decode_package)

            current_output_size //= 2

        # output transformations
        self.logit_head = ConvBlock(
            current_output_size,
            num_classes,
            kernel_size=1
        )
    

        self.depth_head = ConvBlock(
            current_output_size,
            1,
            kernel_size=1
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used in training, takes an image and returns raw logits and raw depth.
        This is what the loss functions use as input.

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            tuple of (torch.FloatTensor, torch.FloatTensor):
                - logits (b, num_classes, h, w)
                - depth (b, h, w)
        """
        # optional: normalizes the input
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # first full convolution
        current_map = self.first_layer(z)
        
        # encoder layers
        encoder_feature_maps = list()
        for encoder in self.encoder_layers:

            encoder_feature_map = encoder(current_map)

            # save these for res connections between decoder and encoder
            encoder_feature_maps.append(encoder_feature_map)
            current_map = encoder_feature_map
        
        # bottleneck
        x = self.bottleneck(current_map)

        zip_decode_iter = zip(
            self.decode_layers,
            reversed(encoder_feature_maps) # iterate through encoder map stack
        )
        # decode layers
        for (fl, upsample), encoder_feature_map in zip_decode_iter:
            
            # concat the first layer with the encoder map
            x = torch.concat([fl(x), encoder_feature_map], dim=1)

            # upsample the concatenation
            x = upsample(x)

        # output transformation heads
        logits = self.logit_head(x)
        raw_depth = self.depth_head(x)

        # reshape the raw depth to remove channel dim
        rd_1 = raw_depth.shape[-2]
        rd_2 = raw_depth.shape[-1]
        raw_depth = raw_depth.reshape(-1, rd_1, rd_2)

        return logits, raw_depth

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used for inference, takes an image and returns class labels and normalized depth.
        This is what the metrics use as input (this is what the grader will use!).

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            tuple of (torch.LongTensor, torch.FloatTensor):
                - pred: class labels {0, 1, 2} with shape (b, h, w)
                - depth: normalized depth [0, 1] with shape (b, h, w)
        """
        logits, raw_depth = self(x)
        pred = logits.argmax(dim=1)

        # Optional additional post-processing for depth only if needed
        depth = raw_depth

        return pred, depth


MODEL_FACTORY = {
    "classifier": Classifier,
    "detector": Detector,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Args:
        model: torch.nn.Module

    Returns:
        float, size in megabytes
    """

    model_size_mb = sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
    print(f"Loaded classification model with size {model_size_mb} MB")

    return model_size_mb


def debug_model(batch_size: int = 1):
    """
    Test your model implementation

    Feel free to add additional checks to this function -
    this function is NOT used for grading
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_batch = torch.rand(batch_size, 3, 64, 64).to(device)

    print(f"Input shape: {sample_batch.shape}")

    model = load_model("classifier", in_channels=3, num_classes=6).to(device)
    output = model(sample_batch)

    # should output logits (b, num_classes)
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    debug_model()
