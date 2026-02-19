# torch.cond vs torch.switch Implementation Comparison

This document describes **intentional** differences between the two implementations. The implementations are aligned wherever possible and differ only where semantics require (e.g. pred vs index, 2 branches vs N branches).

---

## 1. HOP layer (`cond.py` vs `switch.py`)

| Aspect | cond | switch |
|--------|------|--------|
| Op name | `"cond"` | `"switch"` |
| Args | `(pred, true_fn, false_fn, operands)` | `(index, branches, operands)` |
| Constant specialization | `if pred: true_fn else false_fn` | `branches[idx](*operands)` with bounds check |
| gen_schema | 2 graphs (then_gm, else_gm), 2 add_arg for branches | N graphs in loop, N add_arg "branch{idx}" |
| trace_* | true_graph, false_graph; 2 register_module; args = (pred, true_graph, false_graph, operands) | branch_graphs list; N register_module with prefix_{bi}; args = (index, tuple(branch_nodes), operands) |
| Dense impl | `if pred: return true_fn(*operands) else return false_fn(*operands)` | `idx = _index_to_int(index)`; bounds check; `return branches[idx](*operands)` |
| Autograd forward | ctx._pred, ctx._true_bw_fn, ctx._false_bw_fn | ctx._index, ctx._branch_bw_fns (list) |
| Autograd backward | true_bw_gm, false_bw_gm; cond_op(ctx._pred, ...); return `None, None, None, *fill_none_with_masks` | branch_bw_gms list; switch_op(ctx._index, tuple(branch_bw_gms), args); return `(None, None) + tuple(...)` |
| FakeTensorMode | Runs both branches, merges with _merge_output | Runs all branches, merges with _merge_output (from cond) |
| Public API | Docstring, _validate_input, constant-pred warning | Docstring (Args/Restrictions), _validate_input, constant-index warning |
| py_functionalize_impl | cond_func | switch_func |
| py_impl(Vmap) | cond_batch_rule | switch_batch_rule |

---

## 2. Dynamo (`higher_order_ops.py`)

| Aspect | CondHigherOrderVariable | SwitchHigherOrderVariable |
|--------|-------------------------|---------------------------|
| _HOP_NAME | `"torch.cond"` | `"torch.switch"` |
| Kwargs pop | `["pred", "true_fn", "false_fn", "operands"]` | `["index", "branches", "operands"]` |
| Args count | 4 | 3 |
| Constant specialize | pred constant → one of two branches | index constant + in range → one of N branches |
| Predicate/index check | "improper predicate", "bool or boolean tensor with single item" | "improper index", "int or 0-dim int tensor" |
| Branches | two callables (true_fn, false_fn) | branches_var.unpack_var_sequence; num_branches; empty check |
| speculate_branch | branch: bool → args[1] or args[2] | branch_idx: int → branches_seq[branch_idx] |
| Speculation | speculate_branch(True), speculate_branch(False); true_nn_modules, false_nn_modules | Loop speculate_branch(i); branch_nn_modules list |
| Output spec check | same_spec = true_spec.treespec == false_spec.treespec | ref_spec vs results[i][1].treespec for i in 1..N-1 |
| Merge | _merge_graph_inputs once (true_graph, false_graph, ...) | N-way: merge g0 with g1, then g0 with g2, ...; then propagate to g1..gN-1 |
| Single-branch | N/A | combined_inputs from lifted0 + placeholder order |
| install_subgraph | "cond_true", "cond_false" | "switch_branch_{i}" for i in range(num_branches) |
| p_args | (pred, true_node, false_node, tuple(...)) | (index, tuple(branch_nodes), flat_operands) |
| HOP call | torch.ops.higher_order.cond | torch.ops.higher_order.switch |
| speculate_subgraph description | self._HOP_NAME | f"{self._HOP_NAME} branch {branch_idx}" |

---

## 3. Inductor IR (`ir.py`)

| Aspect | Conditional | Switch |
|--------|-------------|--------|
| Attributes | predicate, operands, true_subgraph, false_subgraph, outputs | index, operands, branch_subgraphs (sequence), outputs |
| create() args | (predicate, true_fn, false_fn, operands) | (index, branch_fns, operands) |
| Subgraph iteration | for subgraph in (true_fn, false_fn) | for subgraph in branch_fns |
| Output validation | true_outputs, false_outputs; _has_aliased_buffers "true_fn"/"false_fn" | ref_outputs from branch_fns[0]; loop _has_aliased_buffers "branch {i}" |
| Device | from operands + [predicate] | from operands + [index_ir] |
| codegen | wrapper.codegen_conditional(self) | wrapper.codegen_switch(self) |

Both share: realize_input, fake_operands, _require_exact_strides, make_subgraph + run + graph_outputs, MultiOutputLayout, MultiOutput with FixedLayout, unbacked_bindings, get_unbacked_symbol_defs.

---

## 4. Inductor lowering (`lowering.py`)

| cond | switch |
|------|--------|
| cond(pred, true_fn, false_fn, operands) | switch(index, branches, operands); branch_list = list(branches) or branches |
| ir.Conditional.create(pred, true_fn, false_fn, operands) | ir.Switch.create(index, branch_list, list(operands)) |
| disable_cudagraphs_reason | "control flow operator: torch.cond." | "control flow operator: torch.switch." |

Structure is the same (triton check, create, map TensorBox.create).

---

## 5. Wrapper codegen (`wrapper.py`, `wrapper_fxir.py`)

| Aspect | cond | switch |
|--------|------|--------|
| Line type | ConditionalLine(node: ir.Conditional) | SwitchLine(node: ir.Switch) |
| codegen_conditional / codegen_switch | predicate; if/else; EnterSubgraphLine for true/false | index_ref; if/elif index_ref == i for each branch |
| _generate_* | get_subgm_attr true_subgraph, false_subgraph; predicate, operands; call_function(cond, ...) | branch_subgms from branch_subgraphs; index_node, operands; call_function(switch, ...) |
| Python path | N/A | Loop if/elif, EnterSubgraphLine, codegen_subgraph/by_inlining, ExitSubgraphLine |

Proxy impl name: both use `inner`.

---

## 6. CUDA graph (`cudagraph_conditional_nodes.py`)

| Aspect | if_else_node | switch_node |
|--------|--------------|-------------|
| Input | pred (bool tensor) | index (int tensor) |
| Branches | 2: (pred, true_fn), (!pred, false_fn) | N: for each i, pred = (index == i).reshape([]) |
| Native API | _if_body(pred) only | hasattr(begin_capture_to_switch_node) → native switch node; else N× _if_body |
| Output merge | Copy else into if branch outputs | Copy branch i into outs[0] when i>=1 |
