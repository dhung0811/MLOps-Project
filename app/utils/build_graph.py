"""
Utility functions extracted from codebuggy-infer.ipynb
"""
import re, joblib
import torch
import numpy as np
import javalang
from typing import List, Dict, Tuple, Optional

from utils.gumtree_diff import GumTreeDiff, EditType

GUMTREE_PATH = "./resources/gumtree-4.0.0-beta4/bin/gumtree"
NODE_TYPE_PATH = "./resources/node_type_to_id.joblib"

gumtree_diff = GumTreeDiff(gumtree_path=GUMTREE_PATH)
print(f"✓ GumTree initialized")

node_type_to_id = joblib.load(NODE_TYPE_PATH)
print(f"✓ Loaded node type mapping: {len(node_type_to_id)} types")


# Constants
STATEMENT_NODES = {
    "StatementExpression",
    "ReturnStatement",
    "IfStatement",
    "ForStatement",
    "WhileStatement",
    "DoStatement",
    "SwitchStatement",
    "TryStatement",
    "ThrowStatement",
    "BreakStatement",
    "ContinueStatement",
    "BlockStatement",
}

WRAP_TEMPLATE = """public class Dummy {{
    {method_code}
}}"""

RELATIONS = [
    "AST_CHILD",
    "AST_PARENT",
    "CFG_NEXT",
    "CFG_TRUE",
    "CFG_FALSE",
    "CFG_LOOP",
    "DEF_USE",
    "USE_DEF",
    "DIFF_PARENT",
    "DIFF_SIBLING",
]

RELATION_TO_ID = {r: i for i, r in enumerate(RELATIONS)}


def wrap_method(method_code: str) -> str:
    return WRAP_TEMPLATE.format(method_code=method_code)


def iter_children(node):
    for child in node.children:
        if child is None:
            continue
        if isinstance(child, list):
            for item in child:
                if item is not None:
                    yield item
        else:
            yield child


def get_node_label(node) -> Optional[str]:
    if hasattr(node, "name") and node.name:
        return str(node.name)
    if hasattr(node, "member") and node.member:
        return str(node.member)
    if hasattr(node, "value") and node.value is not None:
        return str(node.value)
    if hasattr(node, "operator") and node.operator:
        return str(node.operator)
    if hasattr(node, "type") and isinstance(node.type, str):
        return str(node.type)
    return None


def compute_line_offsets(code: str) -> List[int]:
    offsets = [0]
    for idx, ch in enumerate(code):
        if ch == "\n":
            offsets.append(idx + 1)
    return offsets


def line_col_to_offset(code: str, line: int, col: int) -> Optional[int]:
    if line <= 0 or col <= 0:
        return None
    line_offsets = compute_line_offsets(code)
    if line > len(line_offsets):
        return None
    return line_offsets[line - 1] + (col - 1)


def offset_to_line_col(code: str, offset: int) -> Tuple[int, int]:
    if offset < 0:
        return (1, 1)
    line_offsets = compute_line_offsets(code)
    line = 1
    for i, start in enumerate(line_offsets, 1):
        if start <= offset:
            line = i
        else:
            break
    col = offset - line_offsets[line - 1] + 1
    return (line, col)


def build_ast_graph(code: str):
    """Build AST graph from Java code"""
    try:
        tree = javalang.parse.parse(wrap_method(code))
    except Exception as e:
        # If parsing fails, try to parse as-is (might already be wrapped)
        try:
            tree = javalang.parse.parse(code)
        except Exception as e2:
            raise ValueError(f"Failed to parse Java code: {e}. Also failed without wrapping: {e2}")

    nodes = []
    parents = []
    children = []
    id_to_index = {}

    def is_wrapper(node) -> bool:
        if isinstance(node, javalang.tree.CompilationUnit):
            return True
        if isinstance(node, javalang.tree.ClassDeclaration) and getattr(node, "name", None) == "Dummy":
            return True
        return False

    def visit(node, parent_idx: Optional[int]):
        if not isinstance(node, javalang.tree.Node):
            return
        if is_wrapper(node):
            for child in iter_children(node):
                visit(child, parent_idx)
            return

        idx = len(nodes)
        id_to_index[id(node)] = idx

        label = get_node_label(node)
        line = None
        col = None
        if getattr(node, "position", None):
            line = node.position.line
            col = node.position.column
        start_pos = line_col_to_offset(code, line, col) if line and col else None
        end_pos = start_pos + len(label) if (start_pos is not None and label) else start_pos

        nodes.append(
            {
                "raw": node,
                "node_type": node.__class__.__name__,
                "label": label,
                "line": line,
                "col": col,
                "start_pos": start_pos,
                "end_pos": end_pos,
            }
        )
        parents.append(parent_idx)
        children.append([])
        if parent_idx is not None:
            children[parent_idx].append(idx)

        for child in iter_children(node):
            visit(child, idx)

    visit(tree, None)
    
    if not nodes:
        raise ValueError("No nodes found in AST. Code might be invalid.")

    edges = []
    edge_types = []

    def add_edge(src: int, dst: int, rel: str):
        edges.append((src, dst))
        edge_types.append(RELATION_TO_ID[rel])

    # AST edges
    for parent_idx, child_list in enumerate(children):
        for child_idx in child_list:
            add_edge(parent_idx, child_idx, "AST_CHILD")
            add_edge(child_idx, parent_idx, "AST_PARENT")

    # CFG edges
    for parent_idx, child_list in enumerate(children):
        stmt_children = [c for c in child_list if nodes[c]["node_type"] in STATEMENT_NODES]
        for a, b in zip(stmt_children, stmt_children[1:]):
            add_edge(a, b, "CFG_NEXT")

    for idx, node in enumerate(nodes):
        node_type = node["node_type"]
        raw = node["raw"]

        if node_type == "IfStatement":
            then_node = getattr(raw, "then_statement", None)
            else_node = getattr(raw, "else_statement", None)
            then_idx = id_to_index.get(id(then_node))
            else_idx = id_to_index.get(id(else_node))
            if then_idx is not None:
                add_edge(idx, then_idx, "CFG_TRUE")
            if else_idx is not None:
                add_edge(idx, else_idx, "CFG_FALSE")

        if node_type in {"ForStatement", "WhileStatement", "DoStatement"}:
            body = getattr(raw, "body", None)
            body_idx = id_to_index.get(id(body))
            if body_idx is not None:
                add_edge(idx, body_idx, "CFG_LOOP")

    # DEF-USE edges
    last_def = {}

    def extract_assigned_name(raw_node) -> Optional[str]:
        target = getattr(raw_node, "expressionl", None) or getattr(raw_node, "left", None)
        if target is None:
            return None
        if hasattr(target, "member") and target.member:
            return str(target.member)
        if hasattr(target, "name") and target.name:
            return str(target.name)
        return None

    for idx, node in enumerate(nodes):
        node_type = node["node_type"]
        label = node["label"]
        raw = node["raw"]

        if node_type in {"VariableDeclarator", "FormalParameter"} and label:
            last_def[label] = idx

        if node_type == "Assignment":
            assigned = extract_assigned_name(raw)
            if assigned:
                last_def[assigned] = idx

        if node_type == "MemberReference" and label:
            if label in last_def:
                def_idx = last_def[label]
                add_edge(def_idx, idx, "DEF_USE")
                add_edge(idx, def_idx, "USE_DEF")

    return nodes, parents, children, edges, edge_types


def compute_code_embeddings(code: str, nodes: List[Dict], children: List[List[int]], tokenizer, encoder, device):
    """Compute code embeddings using GraphCodeBERT"""
    enc = tokenizer(
        code,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        max_length=512,
    )
    offsets = enc.pop("offset_mapping")[0].tolist()
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.inference_mode():
        outputs = encoder(**enc)
    token_embs = outputs.last_hidden_state[0].cpu()

    num_nodes = len(nodes)
    node_embs = torch.zeros((num_nodes, token_embs.shape[-1]), dtype=torch.float)

    for idx, node in enumerate(nodes):
        start = node["start_pos"]
        end = node["end_pos"]
        if start is None or end is None or start == end:
            continue
        token_idxs = [
            t for t, (s, e) in enumerate(offsets)
            if not (s == 0 and e == 0) and s < end and e > start
        ]
        if token_idxs:
            node_embs[idx] = token_embs[token_idxs].mean(dim=0)

    # Propagate embeddings to parent nodes
    for idx in reversed(range(num_nodes)):
        if torch.all(node_embs[idx] == 0) and children[idx]:
            child_embs = [node_embs[c] for c in children[idx] if torch.any(node_embs[c] != 0)]
            if child_embs:
                node_embs[idx] = torch.stack(child_embs).mean(dim=0)

    return node_embs


def format_top_predictions(node_probs: np.ndarray, nodes: List[Dict], code: str, top_k: int = 10) -> str:
    """Format top predictions as readable string"""
    lines = code.splitlines()
    ranked = sorted(range(len(node_probs)), key=lambda i: node_probs[i], reverse=True)
    
    output = []
    output.append(f"\nTop {top_k} Bug Predictions:")
    output.append("=" * 80)
    
    for rank, idx in enumerate(ranked[:top_k], 1):
        node = nodes[idx]
        line = node.get("line")
        col = node.get("col")
        label = node.get("label") or ""
        node_type = node.get("node_type")
        prob = float(node_probs[idx])

        if prob < 0.1:
            continue    

        line_text = ""
        if line is not None and 1 <= line <= len(lines):
            line_text = lines[line - 1].strip()

        # Handle None values for line and col
        line_str = f"{line:3d}" if line is not None else "  -"
        col_str = f"{col:3d}" if col is not None else "  -"
        
        output.append(
            f"{rank:2d}. prob={prob:.4f} | {node_type:20s} | {label:15s} | "
            f"line={line_str} col={col_str}\n    {line_text}"
        )
    
    return "\n".join(output)


def get_node_type_id(node_type: str) -> int:
    return node_type_to_id.get(node_type, node_type_to_id["UNK"])


def add_diff_edges(children, parents, action_map, edges, edge_types):
    changed_nodes = set(action_map.keys())
    for idx in changed_nodes:
        parent = parents[idx]
        if parent is not None:
            edges.append((idx, parent))
            edge_types.append(RELATION_TO_ID["DIFF_PARENT"])
            for sibling in children[parent]:
                if sibling == idx:
                    continue
                edges.append((idx, sibling))
                edge_types.append(RELATION_TO_ID["DIFF_SIBLING"])
                edges.append((sibling, idx))
                edge_types.append(RELATION_TO_ID["DIFF_SIBLING"])


def build_diff_features(action_map: dict[int, EditType], subtree_changed: list[int], num_nodes: int):
    diff_feats = []
    labels = []
    for idx in range(num_nodes):
        action = action_map.get(idx)
        is_diff = 1 if action is not None else 0
        action_none = 1 if action is None else 0
        action_update = 1 if action == EditType.UPDATE else 0
        action_delete = 1 if action == EditType.DELETE else 0
        action_move = 1 if action == EditType.MOVE else 0
        diff_feats.append([
            is_diff,
            action_none,
            action_update,
            action_delete,
            action_move,
            subtree_changed[idx],
        ])
        labels.append(1 if action in {EditType.UPDATE, EditType.DELETE} else 0)
    return torch.tensor(diff_feats, dtype=torch.float), torch.tensor(labels, dtype=torch.long)


def match_actions_to_nodes(code: str, nodes: list[dict], parents: list[int | None], actions):
    action_map: dict[int, EditType] = {}

    for action in actions:
        node_type = action.node.node_type
        label = action.node.label
        pos = action.node.position
        line = None
        col = None
        if pos is not None:
            line, col = offset_to_line_col(code, pos[0])

        candidates = [i for i, n in enumerate(nodes) if n["node_type"] == node_type]
        if label:
            label_candidates = [i for i in candidates if nodes[i]["label"] == label]
            if label_candidates:
                candidates = label_candidates
        if line:
            line_candidates = [i for i in candidates if nodes[i]["line"] == line]
            if line_candidates:
                candidates = line_candidates

        if not candidates:
            continue

        if col is not None:
            candidates.sort(key=lambda i: abs((nodes[i]["col"] or col) - col))

        matched_idx = candidates[0]
        action_map[matched_idx] = action.action_type

    subtree_changed = [0] * len(nodes)
    for idx in action_map.keys():
        cur = parents[idx]
        while cur is not None:
            subtree_changed[cur] = 1
            cur = parents[cur]

    return action_map, subtree_changed


def build_graph_parts(buggy_code: str, fixed_code: str, method_id: str=None):
    nodes, parents, children, edges, edge_types = build_ast_graph(buggy_code)

    diff_result = gumtree_diff.diff(buggy_code, fixed_code)
    action_map, subtree_changed = match_actions_to_nodes(buggy_code, nodes, parents, diff_result.actions)

    diff_feats, labels = build_diff_features(action_map, subtree_changed, len(nodes))
    add_diff_edges(children, parents, action_map, edges, edge_types)

    node_type_ids = torch.tensor([get_node_type_id(n["node_type"]) for n in nodes], dtype=torch.long)

    return {
        "buggy_code": buggy_code,
        "nodes": nodes,
        "parents": parents,
        "children": children,
        "edges": edges,
        "edge_types": edge_types,
        "diff_feats": diff_feats,
        "labels": labels,
        "node_type_ids": node_type_ids,
        "method_id": method_id,
    }
