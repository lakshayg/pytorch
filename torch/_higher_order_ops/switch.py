# mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
import contextlib
import functools
import logging
import warnings
from collections.abc import Callable, Sequence
from typing import Any, Union

import torch
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._C._functorch import (
    _add_batch_dim,
    get_unwrapped,
    is_batchedtensor,
    maybe_get_bdim,
)
from torch._functorch.utils import exposed_in
from torch._higher_order_ops.utils import (
    _maybe_run_with_interpreter,
    check_input_alias_and_mutation_return_outputs,
    create_bw_fn,
    fill_none_with_masks,
    filter_with_masks,
    materialize_as_graph,
    reenter_make_fx,
    save_values_for_backward,
    saved_values,
    unique_graph_id,
    validate_subgraph_args_types,
)
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode, track_tensor_tree
from torch.utils._python_dispatch import _get_current_dispatch_mode


log = logging.getLogger(__name__)


class SwitchOp(HigherOrderOperator):
    def __init__(self):
        super().__init__("switch")

    def __call__(self, index, branches, operands):
        validate_subgraph_args_types(operands)
        return super().__call__(index, branches, operands)

    # pyrefly: ignore [bad-override]
    def gen_schema(self, index, branches, operands):
        from torch._higher_order_ops.schema import HopSchemaGenerator
        from torch._higher_order_ops.utils import materialize_as_graph

        branch_gms: List[torch.fx.GraphModule] = [materialize_as_graph(fn, operands) for fn in branches]
        mutated_inputs = set()
        for gm in branch_gms:
            (
                _,
                _,
                _,
                branch_mutated_inputs,
                branch_outputs,
            ) = check_input_alias_and_mutation_return_outputs(gm)
            mutated_inputs.update(branch_mutated_inputs)

        schema_gen = HopSchemaGenerator(self)
        schema_gen.add_arg("index", index)
        for idx, gm in enumerate(branch_gms):
            schema_gen.add_arg(f"branch{idx}", gm)
        for idx, arg in enumerate(operands):
            schema_gen.add_arg(f"operand{idx}", arg, is_mutated=idx in mutated_inputs)

        for out in branch_outputs: # NOTE: we're using the output from the last branch
            schema_gen.add_output(out)
        schema_gen.add_schema_tree_spec(index, branches, operands)
        return schema_gen.gen_schema()


switch_op = SwitchOp()


@exposed_in("torch")
def switch(
    index: Union[int, torch.Tensor],
    branches: Union[tuple[Callable, ...], list[Callable]],
    operands: Union[tuple, list] = (),
) -> Any:
    r"""
    Selects and runs one of N branch functions by index.

    Equivalent to: branches[index](*operands) with index in [0, len(branches)).

    Args:
        index (Union[int, torch.Tensor]): An int or 0-dim tensor in [0, len(branches)),
          indicating which branch to run.

        branches (Union[tuple[Callable, ...], list[Callable]]): Non-empty sequence of
          callables. Each must accept operands and return the same structure of outputs.

        operands (Tuple of possibly nested dict/list/tuple of torch.Tensor): Inputs to
          the branch functions. Defaults to ().

    Restrictions:
        - index must be an int or a torch.Tensor with a single element (in [0, len(branches))).
        - Each branch must have the same signature as operands and return the same
          output structure (shape, dtype, etc.).
        - Branches cannot have in-place mutations on inputs or global variables.
    """
    if torch.compiler.is_dynamo_compiling():
        return switch_op(index, branches, operands)

    if isinstance(index, int):
        # This is the non-strict export case. Strict export and torch.compile are
        # handled above in dynamo.
        if torch.compiler.is_compiling():
            warnings.warn(
                "Index is a Python constant. When used with torch.switch, it specializes on one of the branches."
                " If you want torch.switch to preserve the branches, please make the predicate an int tensor or a SymInt.",
                UserWarning,
                stacklevel=2,
            )
        # This is the eager case. We can just run the relevant branch
        return branches[index](*operands)

    def _validate_input(index, branches, operands):
        if not isinstance(index, (int, torch.Tensor, torch.SymInt)):
            raise RuntimeError(f"Expected index to be an int or tensor, but got {index}.")

        if isinstance(index, torch.Tensor) and index.numel() != 1:
            raise RuntimeError(
                f"Expected index to be int or single-element tensor, but got {index}."
            )

        index_item = index.item() is isinstance(index, torch.Tensor) else index
        if index_item < 0 or index_item >= len(branches):
            raise RuntimeError(f"switch index must be in [0, {len(branches)}), got {index_item}.")

        if not isinstance(branches, (tuple, list)) or len(branches) == 0:
            raise RuntimeError("Expected branches to be a non-empty tuple or list of callables.")

        for i, branch in enumerate(branches):
            if not callable(branch):
                raise RuntimeError(f"Expected all branches to be callable. branch{i} is not callable.")

        if not isinstance(operands, (tuple, list)) or pytree.tree_any(
            lambda t: not isinstance(t, torch.Tensor), operands
        ):
            raise RuntimeError(
                "Expect operands to be a tuple of possibly nested dict/list/tuple that only "
                f"consists of tensor leaves, but got {operands}."
            )

    _validate_input(index, branches, operands)

    if not torch._dynamo.is_dynamo_supported():
        raise RuntimeError("torch.switch requires dynamo support.")

    # Dynamo is expecting a callable with "__code__" attribute.
    # We cannot directly pass switch_op to it. So we wrap it in a dummy function.
    def _switch_op_wrapper(*args, **kwargs):
        return switch_op(*args, **kwargs)

    from torch._higher_order_ops.utils import setup_compilation_env

    with setup_compilation_env() as backend:
        return torch.compile(_switch_op_wrapper, backend=backend, fullgraph=True)(
            index, branches, operands
        )


def trace_switch(proxy_mode, func_overload, index, branches, operands):
    if not isinstance(operands, (list, tuple)):
        raise AssertionError(
            f"Switch operands must be a list or tuple of tensors and SymInts {operands}"
        )
    if not isinstance(branches, (list, tuple)) or len(branches) == 0:
        raise AssertionError("Switch branches must be a non-empty list or tuple of callables")

    branch_graphs = [reenter_make_fx(branch)(*operands) for branch in branches]

    branch_outs = [list() for _ in branches]
    for i, branch_graph in enumerate(branch_graphs):
        for node in branch_graph.graph.nodes:
            if node.op == "output":
                branch_outs[i].extend(node.args)

    flat_branch_outs = [pytree.arg_tree_leaves(*outs) for outs in branch_outs]
    for i, outs in enumerate(flat_branch_outs):
        if len(flat_branch_outs[0]) != len(outs):
            # TODO define SwitchOpArgsMismatchError
            raise torch._dynamo.exc.SwitchOpArgsMismatchError(
                f"Expected to return same number of outputs from all branches but got:"
                f"\n  branch0 returns {len(flat_branch_outs[0])} item(s)"
                f"\n  branch{i} returns {len(outs)} item(s)"
            )

    uid, _ = unique_graph_id(proxy_mode, prefix="branch0_graph")
    for i, branch_graph in enumerate(branch_graphs):
        proxy_mode.tracer.root.register_module(f"branch{i}_graph_{uid}", branch_graph)

    args = (index, branch_graphs, operands)

    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, args)

    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function", func_overload, proxy_args, {}
    )

    out = func_overload(index, branch_graphs, operands)

    return track_tensor_tree(out, out_proxy, constant=None, tracer=proxy_mode.tracer)


@switch_op.py_impl(DispatchKey.CompositeExplicitAutograd)
def switch_op_dense(index, branches, operands):
    if not all(isinstance(o, (torch.Tensor, int)) for o in operands):
        raise AssertionError(
            f"Dense implementation operands must be a list of tensors and ints {operands}"
        )
    mode = _get_current_dispatch_mode()
    if mode is not None:
        raise AssertionError("Mode should never be enabled for CPU/CUDA key")
    idx = index.item() if isinstance(index, torch.Tensor) else int(index)
    if idx < 0 or idx >= len(branches):
        raise RuntimeError(f"switch index must be in [0, {len(branches)}), got {idx}")
    return branches[idx](*operands)


class SwitchAutogradOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, index, branches, *operands):
        ctx._index = index
        ctx._branch_bw_fns = [
            create_bw_fn(fn, operands) for fn in branches
        ]
        # We snapshot the dispatch keys in forward for materializing the
        # the bw_graph in backward.
        ctx._fw_include_key_set = torch._C._dispatch_tls_local_include_set()
        ctx._fw_exclude_key_set = torch._C._dispatch_tls_local_exclude_set()
        save_values_for_backward(ctx, operands)

        with torch._C._AutoDispatchBelowAutograd():
            return switch_op(index, branches, operands)

    @staticmethod
    def backward(ctx, *flat_grads):
        operands = saved_values(ctx)
        args = operands + flat_grads
        # TODO: we need to materialize the bw graphs because dynamo is unable to
        # trace through the joint function when torch.compile torch.autograd.grad.

        grads_tensor_masks = []

        def create_fn_remove_none(fn):
            @functools.wraps(fn)
            def wrapped(*args):
                nonlocal grads_tensor_masks
                outputs = fn(*args)
                grads_tensor_masks = [
                    bool(isinstance(out, torch.Tensor)) for out in outputs
                ]
                return filter_with_masks(outputs, grads_tensor_masks)

            return wrapped

        branch_bw_gms = [
            materialize_as_graph(
                create_fn_remove_none(fn),
                args,
                ctx._fw_include_key_set,
                ctx._fw_exclude_key_set,
                force_enable_grad=True,
            )
            for fn in ctx._branch_bw_fns
        ]
        grads = switch_op(ctx._index, branch_bw_gms, args)
        return None, None, None, *fill_none_with_masks(grads, grads_tensor_masks)


# Note:
# As long as one of the tensors in pred or operands requires grad,
# all the output would require grad with backward fn set to be the CondAutogradOp.
# This is consistent with autograd.Function's semantic.
@switch_op.py_autograd_impl
def switch_autograd(index, branches, operands):
    return SwitchAutogradOp.apply(index, branches, *operands)


@switch_op.py_impl(ProxyTorchDispatchMode)
def inner(mode, index, branches, operands):
    return trace_switch(mode, switch_op, index, branches, operands)


@switch_op.py_impl(FakeTensorMode)
def switch_fake_tensor_mode(mode, index, branches, operands):
    from torch._higher_order_ops.cond import _merge_output
    # Ignore here, because if you've gotten here but you're not manually
    # tracing the inner graphs, that means that you intend to reuse the graph
    # directly.  Which means the old unbacked symbol bindings are appropriate.
    # This strategy will not work if unbacked symbols can escape.
    ignore_fresh_unbacked = contextlib.nullcontext()
    if mode.shape_env:
        ignore_fresh_unbacked = mode.shape_env.ignore_fresh_unbacked_symbols()

    with mode, ignore_fresh_unbacked:
        flat_outs_list = []
        ref_spec = None
        for branch in branches:
            out = branch(*operands)
            flat, spec = pytree.tree_flatten(out)
            if ref_spec is not None and spec != ref_spec:
                raise RuntimeError(
                    "Unmatched output spec from torch.switch branches: "
                    f"branch 0 tree_spec {ref_spec} vs tree_spec {spec}."
                )
            ref_spec = spec
            flat_outs_list.append(flat)

    num_outputs = len(flat_outs_list[0])
    merged = []
    for j in range(num_outputs):
        m = flat_outs_list[0][j]
        for i in range(1, len(branches)):
            m = _merge_output(m, flat_outs_list[i][j], mode)
        merged.append(m)
    return pytree.tree_unflatten(merged, ref_spec)


@switch_op.py_functionalize_impl
def switch_func(ctx, index, branches, inputs):
    from torch._higher_order_ops.utils import _check_alias_and_mutation

    unwrapped_inputs = ctx.unwrap_tensors(inputs)
    unwrapped_index = ctx.unwrap_tensors(index)
    with ctx.redispatch_to_next():
        functional_branches = [
            ctx.functionalize(_maybe_run_with_interpreter(fn)) for fn in branches
        ]
        pre_dispatch = hasattr(ctx, "mode") and ctx.mode.pre_dispatch
        for i, branch in enumerate(branches):
            _check_alias_and_mutation(
                branch, unwrapped_inputs, f"switch_branch_{i}", pre_dispatch
            )
        switch_return = switch_op(
            unwrapped_index, tuple(functional_branches), unwrapped_inputs
        )
        return ctx.wrap_tensors(switch_return)


@switch_op.py_impl(torch._C._functorch.TransformType.Vmap)
def switch_batch_rule(interpreter, index, branches, inputs):
    if not isinstance(inputs, (list, tuple)):
        raise AssertionError(
            f"Switch inputs must be a list or tuple of tensors, got {type(inputs)}"
        )
    if not all(isinstance(i, torch.Tensor) for i in inputs):
        raise AssertionError(
            f"Switch inputs must be a list of tensors, got {[type(i) for i in inputs]}"
        )

    index_is_batched = isinstance(index, torch.Tensor) and is_batchedtensor(index)
    index_ = get_unwrapped(index) if index_is_batched else index

    tensors, in_dims = zip(
        *[
            (get_unwrapped(t), maybe_get_bdim(t)) if is_batchedtensor(t) else (t, None)
            for t in inputs
        ]
    )

    if index_is_batched:
        tensors = (index_,) + tensors
        in_dims = (0,) + in_dims

        def fn(idx, *args):
            branch_outs = [branch(*args) for branch in branches]
            batch_size = idx.shape[0]
            arange = torch.arange(batch_size, device=idx.device)
            if isinstance(branch_outs[0], tuple):
                return tuple(
                    torch.stack([bo[j] for bo in branch_outs], 0)[idx, arange]
                    for j in range(len(branch_outs[0]))
                )
            stacked = torch.stack(branch_outs, 0)
            return stacked[idx, arange]

        with interpreter.lower():
            result = torch.vmap(fn, in_dims=in_dims)(*tensors)
    else:
        idx = index_.item() if isinstance(index_, torch.Tensor) else int(index_)
        vmapped_branch = torch.vmap(branches[idx], in_dims=in_dims)
        with interpreter.lower():
            result = vmapped_branch(*tensors)

    if not isinstance(result, tuple):
        result = (result,)
    lvl = interpreter.level()
    return tuple(_add_batch_dim(r, 0, lvl) for r in result)
