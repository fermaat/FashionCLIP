import torch

from typing import Callable, Optional # , Union
from torch.nn.modules.distance import PairwiseDistance
from torch.nn.modules.loss import _Loss
from torch._prims_common import TensorLikeType
from torch import Tensor


def _apply_loss_reduction(loss: TensorLikeType, reduction: str) -> TensorLikeType:
    if reduction == "sum":
        return torch.sum(loss)
    elif reduction == "mean":
        return torch.mean(loss)
    else:  # reduction == "none"
        return loss


def _check_reduction_value(reduction: str):
    if reduction not in ("mean", "sum", "none"):
        raise ValueError(f"{reduction} is not a valid value for reduction")


def _triplet_semipositive_margin_with_distance_loss(
    anchor: TensorLikeType,
    positive: TensorLikeType,
    negative: TensorLikeType,
    semipositive: TensorLikeType,
    *,
    distance_function: Optional[
        Callable[[TensorLikeType, TensorLikeType], TensorLikeType]
    ] = None,
    margin: float = 1.0,
    swap: bool = False,
    reduction: str = "mean",
) -> TensorLikeType:
    _check_reduction_value(reduction)

    a_dim = anchor.ndim
    p_dim = positive.ndim
    n_dim = negative.ndim
    assert a_dim == p_dim and p_dim == n_dim
    # torch._check(
    #     a_dim == p_dim and p_dim == n_dim,
    #     lambda: (
    #         f"The anchor, positive, and negative tensors are expected to have "
    #         f"the same number of dimensions, but got: anchor {a_dim}D, "
    #         f"positive {p_dim}D, and negative {n_dim}D inputs"
    #     ),
    # )

    if distance_function is None:
        distance_function = torch.pairwise_distance

    dist_pos = distance_function(anchor, positive)
    dist_neg = distance_function(anchor, negative)
    # The distance swap is described in the paper "Learning shallow
    # convolutional feature descriptors with triplet losses" by V. Balntas, E.
    # Riba et al.  If True, and if the positive example is closer to the
    # negative example than the anchor is, swaps the positive example and the
    # anchor in the loss computation.
    if swap:
        # TODO: how does the seimpos fit in here?
        dist_swap = distance_function(positive, negative)
        dist_neg = torch.minimum(dist_neg, dist_swap)
    # d+ = d(a, +) + d_lev(a, +) i.e: distance = distance function + levenstein distance
    loss = torch.clamp_min(margin + dist_pos - dist_neg + semipositive, 0)
    return _apply_loss_reduction(loss, reduction)


class TripletSemiPosMarginWithDistanceLoss(_Loss):
    r"""Creates a criterion that measures the triplet loss given input
    tensors :math:`a`, :math:`p`, and :math:`n` (and :math:`sp`)(representing anchor,
    positive, negative and semipos examples, respectively), and a nonnegative,
    real-valued function ("distance function") used to compute the relationship
    between the anchor and positive example ("positive distance") and the
    anchor and negative example ("negative distance").

    The unreduced loss (i.e., with :attr:`reduction` set to ``'none'``)
    can be described as:

    .. math::
        \ell(a, p, n) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_i = \max \{d(a_i, p_i) + sp - d(a_i, n_i) + {\rm margin}, 0\}

    where :math:`N` is the batch size; :math:`d` is a nonnegative, real-valued function
    quantifying the closeness of two tensors, referred to as the :attr:`distance_function`;
    and :math:`margin` is a nonnegative margin representing the minimum difference
    between the positive and negative distances that is required for the loss to
    be 0.  :math:`sp` is the semipos distance between anchor and positive (for a semipositive example).
    The input tensors have :math:`N` elements each and can be of any shape
    that the distance function can handle.

    If :attr:`reduction` is not ``'none'``
    (default ``'mean'``), then:

    .. math::
        \ell(x, y) =
        \begin{cases}
            \operatorname{mean}(L), &  \text{if reduction} = \text{`mean';}\\
            \operatorname{sum}(L),  &  \text{if reduction} = \text{`sum'.}
        \end{cases}

    See also :class:`~torch.nn.TripletMarginLoss`, which computes the triplet
    loss for input tensors using the :math:`l_p` distance as the distance function.

    Args:
        distance_function (Callable, optional): A nonnegative, real-valued function that
            quantifies the closeness of two tensors. If not specified,
            `nn.PairwiseDistance` will be used.  Default: ``None``
        margin (float, optional): A nonnegative margin representing the minimum difference
            between the positive and negative distances required for the loss to be 0. Larger
            margins penalize cases where the negative examples are not distant enough from the
            anchors, relative to the positives. Default: :math:`1`.
        swap (bool, optional): Whether to use the distance swap described in the paper
            `Learning shallow convolutional feature descriptors with triplet losses` by
            V. Balntas, E. Riba et al. If True, and if the positive example is closer to the
            negative example than the anchor is, swaps the positive example and the anchor in
            the loss computation. Default: ``False``.
        reduction (str, optional): Specifies the (optional) reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``


    Shape:
        - Input: :math:`(N, *)` where :math:`*` represents any number of additional dimensions
          as supported by the distance function.
        - Output: A Tensor of shape :math:`(N)` if :attr:`reduction` is ``'none'``, or a scalar
          otherwise.

    Examples::

    >>> # Initialize embeddings
    >>> embedding = nn.Embedding(1000, 128)
    >>> anchor_ids = torch.randint(0, 1000, (1,))
    >>> positive_ids = torch.randint(0, 1000, (1,))
    >>> negative_ids = torch.randint(0, 1000, (1,))
    >>> semipos = torch.randint(0, 1000, (1,))
    >>> anchor = embedding(anchor_ids)
    >>> positive = embedding(positive_ids)
    >>> negative = embedding(negative_ids)
    >>>
    >>> # Built-in Distance Function
    >>> triplet_loss = \
    >>>     TripletSemiPosMarginWithDistanceLoss(distance_function=nn.PairwiseDistance())
    >>> output = triplet_loss(anchor, positive, negative)
    >>> output.backward()
    >>>
    >>> # Custom Distance Function
    >>> def l_infinity(x1, x2):
    >>>     return torch.max(torch.abs(x1 - x2), dim=1).values
    >>>
    >>> # xdoctest: +SKIP("FIXME: Would call backwards a second time")
    >>> triplet_loss = (
    >>>     TripletSemiPosMarginWithDistanceLoss(distance_function=l_infinity, margin=1.5))
    >>> output = triplet_loss(anchor, positive, negative)
    >>> output.backward()
    >>>
    >>> # Custom Distance Function (Lambda)
    >>> triplet_loss = (
    >>>     TripletSemiPosMarginWithDistanceLoss(
    >>>         distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y)))
    >>> output = triplet_loss(anchor, positive, negative)
    >>> output.backward()

    Reference:
        V. Balntas, et al.: Learning shallow convolutional feature descriptors with triplet losses:
        http://www.bmva.org/bmvc/2016/papers/paper119/index.html
    """
    __constants__ = ['margin', 'swap', 'reduction']
    margin: float
    swap: bool

    def __init__(self, *, distance_function: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
                 margin: float = 1.0, swap: bool = False, reduction: str = 'mean'):
        super().__init__(size_average=None, reduce=None, reduction=reduction)
        self.distance_function: Optional[Callable[[Tensor, Tensor], Tensor]] = \
            distance_function if distance_function is not None else PairwiseDistance()
        # distance_function=lambda x, y: torch.pairwise_distance(x, y, p, eps)
        self.margin = margin
        self.swap = swap

    def forward(self, anchor: Tensor, positive: Tensor, negative: Tensor, semipos: Tensor) -> Tensor:
        return _triplet_semipositive_margin_with_distance_loss(anchor, positive, negative, semipos,
                                                               distance_function=self.distance_function,
                                                               margin=self.margin, swap=self.swap, reduction=self.reduction)
