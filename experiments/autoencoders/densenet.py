import torch.nn as nn


class DenseNet(nn.Module):
    """Implements a fully connected encoder-decoder network."""

    def __init__(
        self, encoder_widths, decoder_widths, act_fn=nn.ReLU(), out_fn=None
    ):
        super(DenseNet, self).__init__()

        assert (
            encoder_widths[-1] == decoder_widths[0]
        ), "encoder output and decoder input dims must match"

        enc_layers = {}
        for k in range(len(encoder_widths) - 1):
            enc_layers[f"enc_layer_{k}"] = nn.Linear(
                encoder_widths[k], encoder_widths[k + 1]
            )
        self.enc_layers = nn.ModuleDict(enc_layers)

        dec_layers = {}
        for k in range(len(decoder_widths) - 1):
            dec_layers[f"dec_layer_{k}"] = nn.Linear(
                decoder_widths[k], decoder_widths[k + 1]
            )
        self.dec_layers = nn.ModuleDict(dec_layers)

        self.act_fn = act_fn
        self.out_fn = out_fn

    def forward(self, x):
        for k in range(len(self.enc_layers)):
            x = self.enc_layers[f"enc_layer_{k}"](x)
            x = x if k == len(self.enc_layers) - 1 else self.act_fn(x)

        for k in range(len(self.dec_layers)):
            x = self.dec_layers[f"dec_layer_{k}"](x)
            if k == len(self.dec_layers) - 1:
                x = x if self.out_fn is None else self.out_fn(x)
            else:
                x = self.act_fn(x)
        return x
