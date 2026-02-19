# mypy: allow-untyped-defs
from collections.abc import Generator
from contextlib import contextmanager

import torch
import torch.utils._pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode


class CUDAGraphCaptureControlFlowOpDispatchMode(TorchDispatchMode):
    @classmethod
    def ignore_compile_internals(cls) -> bool:
        return True

    def __init__(
        self,
    ) -> None:
        self.supports_higher_order_operators = True
        super().__init__()

    def __torch_dispatch__(
        self,
        func,
        types,
        args=(),
        kwargs=None,
    ):
        if func is torch.ops.higher_order.cond:
            with self:
                return if_else_node(*args)
        if func is torch.ops.higher_order.switch:
            with self:
                return switch_node(*args)
        kwargs = {} if kwargs is None else kwargs
        return func(*args, **kwargs)


class ControlFlowOpWarmupDispatchMode(TorchDispatchMode):
    """The purpose of this TodchDispatchMode is to "warm up" both sides of a torch.cond() statement.

    For data-dependent control flow code, only one side will be
    executed. Therefore, it is not safe to stream capture a
    torch.cond() statement naively, since we don't have a guarantee
    that all ops will have been "warmed up". The clever workaround is
    to use a "relaxed" stream capture whose final cuda graph we throw
    away. This works because stream capture does not actually execute
    any GPU code, and because true_fn and false_fn are both fxgraphs,
    which do not have any CPU side effects.
    """

    @classmethod
    def ignore_compile_internals(cls) -> bool:
        return True

    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.supports_higher_order_operators = True
        self.capture_stream = torch.cuda.Stream()

    def __torch_dispatch__(
        self,
        func,
        types,
        args=(),
        kwargs=None,
    ):
        kwargs = {} if kwargs is None else kwargs

        if func is torch.ops.higher_order.cond:
            if torch.cuda.is_current_stream_capturing():
                with self:
                    return if_else_node(*args)
            else:
                with (
                    torch.cuda.graph(
                        torch.cuda.CUDAGraph(),
                        pool=None,
                        stream=self.capture_stream,
                        capture_error_mode="relaxed",
                    ),
                    self,
                ):
                    if_else_node(*args)
                return func(*args, **kwargs)
        if func is torch.ops.higher_order.switch:
            if torch.cuda.is_current_stream_capturing():
                with self:
                    return switch_node(*args)
            else:
                with (
                    torch.cuda.graph(
                        torch.cuda.CUDAGraph(),
                        pool=None,
                        stream=self.capture_stream,
                        capture_error_mode="relaxed",
                    ),
                    self,
                ):
                    switch_node(*args)
                return func(*args, **kwargs)
        return func(*args, **kwargs)


@contextmanager
def _if_body(pred: torch.Tensor) -> Generator[None, None, None]:
    current_cuda_graph = torch.cuda.CUDAGraph.get_currently_capturing_graph()
    current_cuda_graph.begin_capture_to_if_node(pred)
    try:
        yield
    finally:
        current_cuda_graph.end_capture_to_conditional_node()


def if_else_node(pred: torch.Tensor, true_fn, false_fn, operands):
    if not pred.is_cuda:
        raise ValueError(
            "Conditions must be on a cuda device to use conditional node in cuda graphs"
        )
    outs = []

    for lazy_pred, fn in [
        (lambda: pred, true_fn),
        (lambda: torch.logical_not(pred), false_fn),
    ]:
        with _if_body(lazy_pred()):
            outs.append(fn(*operands))
            if len(outs) == 2:
                for if_out, else_out in zip(
                    pytree.tree_iter(outs[0]), pytree.tree_iter(outs[1])
                ):
                    if_out.copy_(else_out)
    return outs[0]


def switch_node(index: torch.Tensor, branches: tuple, operands):
    if not index.is_cuda:
        raise ValueError(
            "switch index must be on a cuda device to use conditional nodes in cuda graphs"
        )
    branches = tuple(branches)
    current_cuda_graph = torch.cuda.CUDAGraph.get_currently_capturing_graph()
    if hasattr(current_cuda_graph, "begin_capture_to_switch_node"):
        current_cuda_graph.begin_capture_to_switch_node(index, len(branches))
        outs = []
        for i, fn in enumerate(branches):
            if i >= 1:
                current_cuda_graph.begin_capture_to_switch_branch(i)
            try:
                outs.append(fn(*operands))
                if i >= 1:
                    for a, b in zip(
                        pytree.tree_iter(outs[0]), pytree.tree_iter(outs[-1])
                    ):
                        a.copy_(b)
            finally:
                current_cuda_graph.end_capture_to_conditional_node()
        return outs[0]
    outs = []
    for i, fn in enumerate(branches):
        pred = (index == i).reshape([])
        with _if_body(pred):
            outs.append(fn(*operands))
            if len(outs) >= 2:
                for first_out, branch_out in zip(
                    pytree.tree_iter(outs[0]), pytree.tree_iter(outs[-1])
                ):
                    first_out.copy_(branch_out)
    return outs[0]
