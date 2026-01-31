import torch


class StepFunction:
    "Step function defined by points (x, y)"

    def __init__(
        self, x: torch.Tensor, y: torch.Tensor, side: str, domain: tuple = (None, None)
    ):
        """Step function defined with point (x,y)

        Args:
            x (torch.Tensor): x-axis points
            y (torch.Tensor): corresponding y-axis points
            side (str): determine the side of the step, whether to take the left or right value
            domain (tuple, optional): domain of the step function
        """
        self.x = x
        self.y = y
        self._side = side
        domain_lower = domain[0] if domain[0] is not None else torch.min(x).item()
        domain_upper = domain[1] if domain[1] is not None else torch.max(x).item()
        self._domain = (domain_lower, domain_upper)

    @property
    def domain(self):
        return self._domain

    @property
    def side(self):
        return self._side

    def __call__(self, x_eval: torch.Tensor):
        """Evaluate the step function at given values"""

        if torch.any(x_eval < self._domain[0]) or torch.any(x_eval > self._domain[1]):
            out_of_domain = x_eval[
                (x_eval < self._domain[0]) | (x_eval > self._domain[1])
            ]
            raise ValueError(
                f"Input {out_of_domain[0]} is out of the function domain: {self._domain}"
            )

        if self._side == "right":
            indices = torch.bucketize(x_eval, self.x)
        else:
            indices = torch.bucketize(x_eval, self.x, right=True) - 1

        # print(indices)
        output = self.y[indices]
        return output

    def __repr__(self):
        return f"StepFunction(x={self.x!r}, y={self.y!r}, side={self.side})"
