import logging
logging.basicConfig(level=logging.INFO)

import torch
import torch.nn as nn
from CausalSurv.model.mlp import MLP


from CausalSurv.tools import load_config

class embed_LSTM(nn.Module):
    """Adaptation of the C-LSTM (Care LSTM, Pham et al. 2017)
    with added embedding layers for X, P, and output state (sa).
    """

    def __init__(self, 
                 x_input_dim: int, 
                 p_input_dim: int,
                 hidden_length: int,
                 output_length: int,
                 x_embed_dim: int,
                 p_embed_dim: int,
                 mlpx_hidden_units: list[int],
                 mlpp_hidden_units: list[int],
                 mlpsa_hidden_units: list[int],
                 mlpx_dropout: float,
                 mlpp_dropout: float,
                 mlpsa_dropout: float,
                 ):
                 
        """Class constructor

        Args:
            x_input_dim (int): Length of input X
            p_input_dim (int): Length of input P    
            output_sa_length (int): Output state size
            cell_config (dict | str): Configuration dictionary for the LSTM cell or path to a TOML config file
        """
        super().__init__()

        self.x_input_dim = x_input_dim
        self.p_input_dim = p_input_dim
        self.output_length = output_length

        self.x_embed_dim = x_embed_dim
        self.p_embed_dim = p_embed_dim
        self.hidden_length = hidden_length

        self.mlpp_hidden_units = mlpp_hidden_units
        self.mlpx_hidden_units = mlpx_hidden_units
        self.mlpp_dropout = mlpp_dropout
        self.mlpx_dropout = mlpx_dropout

        # Embedding MLPs for X and P
        self.MLPx = MLP(input_dim=self.x_input_dim,
                        output_dim=self.x_embed_dim,
                        n_units=self.mlpx_hidden_units,
                        dropout=self.mlpx_dropout,
                        )
        self.MLPp = MLP(input_dim=self.p_input_dim, 
                        output_dim=self.p_embed_dim,
                        n_units=self.mlpp_hidden_units,
                        dropout=self.mlpp_dropout,
                        )

        # Forget gate
        self.linear_forget_Wxf = nn.Linear(self.x_embed_dim, self.hidden_length, bias=True)
        self.linear_forget_Whf = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.linear_forget_Wdf = nn.Linear(1, self.hidden_length, bias=False)
        self.linear_forget_Wpf = nn.Linear(self.p_embed_dim, self.hidden_length, bias=False)
        self.sigmoid_forget = nn.Sigmoid()

        # Input gate
        self.linear_input_Wxi = nn.Linear(self.x_embed_dim, self.hidden_length, bias=True)
        self.linear_input_Whi = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.sigmoid_input = nn.Sigmoid()

        # Cell memory (candidate g)
        self.linear_cell_Wxg = nn.Linear(self.x_embed_dim, self.hidden_length, bias=True)
        self.linear_cell_Whg = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.activation_cell = nn.Tanh()

        # Output gate
        self.linear_out_Wxo = nn.Linear(self.x_embed_dim, self.hidden_length, bias=True)
        self.linear_out_Who = nn.Linear(self.hidden_length, self.hidden_length, bias=False)
        self.linear_out_Wpo = nn.Linear(self.p_embed_dim, self.hidden_length, bias=False)
        self.sigmoid_out = nn.Sigmoid()

        # Final hidden activation
        self.activation_final = nn.Tanh()

        # Output MLP
        self.mlpsa_hidden_units = mlpsa_hidden_units
        self.mlpsa_dropout = mlpsa_dropout
        self.MLPsa = MLP(input_dim=self.hidden_length, 
                         output_dim=self.output_length,
                         n_units=self.mlpsa_hidden_units,
                         dropout=self.mlpsa_dropout)
        
    def _forget_gate(self, x: torch.Tensor, d: torch.Tensor, h_prev: torch.Tensor, p_prev: torch.Tensor) -> torch.Tensor:
        """Compute the forget gate activation.

        Args:
            x (torch.Tensor): Input tensor for X at current time step
            d (torch.Tensor): Input tensor for time decay (elapsed time since last event)
            h_prev (torch.Tensor): Previous hidden state 
            p_prev (torch.Tensor): Previous cell state

        Returns:
            torch.Tensor: Forget gate activation
        """
        d = d.view(-1, 1)
        # Time decay scaling for the forget gate, as in C-LSTM (Pham et al. 2017):
        # This transformation modulates the forget gate output based on elapsed time 'd',
        # allowing the model to retain less memory as time increases.
        func_d = 1 / torch.log(torch.exp(torch.ones_like(d)) + d)
        f =  func_d * self.sigmoid_forget(
            self.linear_forget_Wxf(x) +
            self.linear_forget_Whf(h_prev) +
            self.linear_forget_Wdf(d) +
            self.linear_forget_Wpf(p_prev)
        )
        return f

    def _input_gate(self, x: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """Compute the input gate activation.

        Args:
            x (torch.Tensor): Input tensor for X at current time step
            h_prev (torch.Tensor): Previous hidden state

        Returns:
            torch.Tensor: Input gate activation
        """
        i =  self.sigmoid_input(
            self.linear_input_Wxi(x) +
            self.linear_input_Whi(h_prev)
        )
        return i

    def _cell_memory_gate(self, i: torch.Tensor, f: torch.Tensor, x: torch.Tensor, h_prev: torch.Tensor, c_prev: torch.Tensor) -> torch.Tensor:
        """Compute the cell memory gate activation.

        Args:
            i (torch.Tensor): Input gate activation
            f (torch.Tensor): Forget gate activation
            x (torch.Tensor): Input tensor for X at current time step
            h_prev (torch.Tensor): Previous hidden state
            c_prev (torch.Tensor): Previous cell state

        Returns:
            torch.Tensor: Cell memory gate activation
        """
        g = self.activation_cell(
            self.linear_cell_Wxg(x) +
            self.linear_cell_Whg(h_prev)
        )
        return g * i + f * c_prev

    def _out_gate(self, x: torch.Tensor, h_prev: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """Compute the output gate activation.

        Args:
            x (torch.Tensor): Input tensor for X at current time step
            h_prev (torch.Tensor): Previous hidden state
            p (torch.Tensor): Previous cell state

        Returns:
            torch.Tensor: Output gate activation
        """
        return self.sigmoid_out(
            self.linear_out_Wxo(x) +
            self.linear_out_Who(h_prev) +
            self.linear_out_Wpo(p)
        )

    def forward(self, XPd, tuple_in):
        """Forward pass of the LSTM cell.
        Args:
            XPd (torch.Tensor): shape (batch, 1, features) or (batch, features)
            tuple_in (tuple): (h_prev, c_prev, p_prev)

        Returns:
            tuple: (sa, h, c, p)
        """
        h_prev, c_prev, p_prev = tuple_in

        if XPd.ndim == 3:
            XPd = XPd[:, 0, :]
        X = XPd[:, :self.x_input_dim]
        P = XPd[:, self.x_input_dim:-1]
        d = XPd[:, -1:]

        x = self.MLPx(X)  
        p = self.MLPp(P) 

        i = self._input_gate(x, h_prev)
        f = self._forget_gate(x, d, h_prev, p_prev)
        c = self._cell_memory_gate(i, f, x, h_prev, c_prev)
        o = self._out_gate(x, h_prev, p)
        h = o * self.activation_final(c)

        sa = self.MLPsa(h)
        return sa, h, c, p


if __name__ == "__main__":
    batch_size = 5
    x_input_dim, p_input_dim = 6, 12
    output_sa_length = 7

    model = embed_LSTM(
        x_input_dim=x_input_dim,
        p_input_dim=p_input_dim,
        hidden_length=16,
        output_length=output_sa_length,
        x_embed_dim=8,
        p_embed_dim=8,
        mlpx_hidden_units=[16, 16],
        mlpp_hidden_units=[16, 16],
        mlpsa_hidden_units=[16, 16],
        mlpx_dropout=0.1,
        mlpp_dropout=0.1,
        mlpsa_dropout=0.1,
    )

    time_steps = 19
    features = x_input_dim + p_input_dim + 1
    XPd = torch.randn(batch_size, time_steps, features)

    h_0 = torch.zeros(batch_size, model.hidden_length)
    c_0 = torch.zeros(batch_size, model.hidden_length)
    p_0 = torch.zeros(batch_size, model.p_embed_dim)
    states = (h_0, c_0, p_0)

    import logging
    logging.basicConfig(level=logging.INFO)

    for t in range(time_steps):
        sa, h_0, c_0, p_0 = model(XPd[:, t, :], states)
        states = (h_0, c_0, p_0)

    logging.info(f"Output sa shape: {sa.shape}, should be ({batch_size}, {output_sa_length})")
    logging.info(f"Final hidden state h shape: {h_0.shape}, should be ({batch_size}, {model.hidden_length})")
    logging.info(f"Final cell state c shape: {c_0.shape}, should be ({batch_size}, {model.hidden_length})")
    logging.info(f"Final p state shape: {p_0.shape}, should be ({batch_size}, {model.p_embed_dim})")