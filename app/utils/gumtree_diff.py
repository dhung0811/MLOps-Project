"""
GumTree Diff module for computing AST differences between buggy and fixed code.

Uses GumTree CLI to extract edit actions (insert, delete, update, move)
that transform the buggy code into the fixed version.
"""

import subprocess
import platform
import tempfile
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

# Use system JAVA_HOME if available, otherwise set default for macOS
env = os.environ.copy()
if 'JAVA_HOME' not in env:
    # Default for macOS with Homebrew
    if platform.system() == 'Darwin':
        env['JAVA_HOME'] = '/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home'
    # For Linux/Docker, it should already be set in environment


class EditType(Enum):
    """Types of edit actions in AST diff"""
    INSERT = "insert"
    DELETE = "delete"
    UPDATE = "update"
    MOVE = "move"
    
    @classmethod
    def from_string(cls, s: str) -> "EditType":
        """Convert string to EditType"""
        mapping = {
            "insert-node": cls.INSERT,
            "insert-tree": cls.INSERT,
            "delete-node": cls.DELETE,
            "delete-tree": cls.DELETE,
            "update-node": cls.UPDATE,
            "move-tree": cls.MOVE,
            "insert": cls.INSERT,
            "delete": cls.DELETE,
            "update": cls.UPDATE,
            "move": cls.MOVE,
        }
        return mapping.get(s.lower(), cls.UPDATE)


@dataclass
class ASTDiffNode:
    """
    Represents a node involved in an edit action.
    
    Attributes:
        node_type: Type of the AST node (e.g., 'SimpleName', 'MethodInvocation')
        label: Value/name of the node
        position: (start_pos, end_pos) in source code
        line: Line number in source
    """
    node_type: str
    label: Optional[str] = None
    position: Optional[Tuple[int, int]] = None
    line: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_type": self.node_type,
            "label": self.label,
            "position": self.position,
            "line": self.line
        }


@dataclass
class EditAction:
    """
    Represents a single edit action from GumTree diff.
    
    Attributes:
        action_type: Type of edit (insert, delete, update, move)
        node: The AST node being edited
        parent: Parent node (for inserts)
        at_position: Position in parent's children (for inserts)
        old_value: Original value (for updates)
        new_value: New value (for updates)
        raw_data: Original raw data from GumTree
    """
    action_type: EditType
    node: ASTDiffNode
    parent: Optional[ASTDiffNode] = None
    at_position: Optional[int] = None
    old_value: Optional[str] = None
    new_value: Optional[str] = None
    raw_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_type": self.action_type.value,
            "node": self.node.to_dict(),
            "parent": self.parent.to_dict() if self.parent else None,
            "at_position": self.at_position,
            "old_value": self.old_value,
            "new_value": self.new_value
        }
    
    def to_string(self) -> str:
        """Get human-readable string representation"""
        parts = [f"[{self.action_type.value.upper()}]"]
        parts.append(self.node.node_type)
        
        if self.node.label:
            parts.append(f"'{self.node.label}'")
        
        if self.action_type == EditType.UPDATE and self.new_value:
            parts.append(f"-> '{self.new_value}'")
        
        if self.parent:
            parts.append(f"@ {self.parent.node_type}")
            
        return " ".join(parts)


@dataclass
class DiffResult:
    """
    Result of computing AST diff between two code versions.
    
    Attributes:
        actions: List of edit actions
        src_nodes: Number of nodes in source AST
        dst_nodes: Number of nodes in destination AST
        edit_distance: Total number of edits
        mappings: Node mappings between source and destination
    """
    actions: List[EditAction]
    src_nodes: int = 0
    dst_nodes: int = 0
    edit_distance: int = 0
    mappings: List[Tuple[str, str]] = field(default_factory=list)
    raw_output: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def has_changes(self) -> bool:
        return len(self.actions) > 0
    
    @property
    def insert_count(self) -> int:
        return sum(1 for a in self.actions if a.action_type == EditType.INSERT)
    
    @property
    def delete_count(self) -> int:
        return sum(1 for a in self.actions if a.action_type == EditType.DELETE)
    
    @property
    def update_count(self) -> int:
        return sum(1 for a in self.actions if a.action_type == EditType.UPDATE)
    
    @property
    def move_count(self) -> int:
        return sum(1 for a in self.actions if a.action_type == EditType.MOVE)
    
    def get_actions_by_type(self, action_type: EditType) -> List[EditAction]:
        return [a for a in self.actions if a.action_type == action_type]
    
    def get_edited_node_types(self) -> List[str]:
        """Get unique node types that were edited"""
        return list(set(a.node.node_type for a in self.actions))
    
    def get_deleted_nodes(self) -> List[ASTDiffNode]:
        """Get all nodes that were deleted (these are the buggy parts)"""
        return [a.node for a in self.actions if a.action_type == EditType.DELETE]
    
    def get_inserted_nodes(self) -> List[ASTDiffNode]:
        """Get all nodes that were inserted (these are the fixes)"""
        return [a.node for a in self.actions if a.action_type == EditType.INSERT]
    
    def to_string(self) -> str:
        """Get human-readable diff summary"""
        lines = [
            f"Edit Actions ({len(self.actions)} total):",
            f"  - Inserts: {self.insert_count}",
            f"  - Deletes: {self.delete_count}",
            f"  - Updates: {self.update_count}",
            f"  - Moves: {self.move_count}",
            "",
            "Actions:"
        ]
        for action in self.actions:
            lines.append(f"  {action.to_string()}")
        return "\n".join(lines)


class GumTreeDiff:
    """
    Wrapper for GumTree CLI to compute AST differences between Java code versions.
    
    GumTree supports the following edit actions:
    - insert: Add new node
    - delete: Remove node
    - update: Change node value (e.g., rename variable)
    - move: Move node to different position
    
    Usage:
        gt = GumTreeDiff()
        result = gt.diff(buggy_code, fixed_code)
        for action in result.actions:
            print(action.to_string())
    """
    
    WRAPPED_CODE_TEMPLATE = """public class Dummy {{
    {method_code}
}}"""
    
    # Default paths to look for GumTree installation
    DEFAULT_PATHS_WINDOWS = [
        'E:\\CodeBuggy\\gumtree-4.0.0-beta4\\bin\\gumtree.bat',
        'E:\\CodeBuggy\\gumtree-4.0.0-beta6\\bin\\gumtree.bat',
        '.\\gumtree\\bin\\gumtree.bat',
        'gumtree.bat',
        'gumtree'
    ]
    
    DEFAULT_PATHS_UNIX = [
        './gumtree/bin/gumtree',
        '/usr/local/bin/gumtree',
        'gumtree'
    ]
    
    def __init__(self, gumtree_path: str = None, timeout: int = 30):
        """
        Initialize GumTree wrapper.
        
        Args:
            gumtree_path: Path to GumTree executable. Auto-detected if None.
            timeout: Timeout in seconds for GumTree commands.
        """
        self.is_windows = platform.system() == 'Windows'
        self.timeout = timeout
        
        if gumtree_path is None:
            gumtree_path = self._auto_detect_path()
        
        self.gumtree_path = gumtree_path
        self._validated = False
    
    def _auto_detect_path(self) -> str:
        """Auto-detect GumTree installation path"""
        paths = self.DEFAULT_PATHS_WINDOWS if self.is_windows else self.DEFAULT_PATHS_UNIX
        
        for path in paths:
            if os.path.exists(path):
                return path
        
        # Return default and let validation fail with helpful message
        return 'gumtree.bat' if self.is_windows else 'gumtree'
    
    def validate(self) -> bool:
        """Validate that GumTree is properly installed and working"""
        if self._validated:
            return True
            
        try:
            result = subprocess.run(
                [self.gumtree_path, "--version"],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=10,
                shell=self.is_windows,
                env=env
            )
            self._validated = result.returncode == 0
            return self._validated
        except Exception:
            return False
    
    def _wrap_as_class(self, method_code: str) -> str:
        """Wrap method code in a dummy class for parsing"""
        return self.WRAPPED_CODE_TEMPLATE.format(method_code=method_code)
    
    def _parse_tree_string(self, tree_str: str) -> ASTDiffNode:
        """Parse GumTree tree string format into ASTDiffNode"""
        # Format: "NodeType: label [pos, len]" or "NodeType [pos, len]"
        parts = tree_str.strip()
        
        node_type = parts
        label = None
        position = None
        
        # Extract position if present
        if '[' in parts and ']' in parts:
            pos_start = parts.rfind('[')
            pos_end = parts.rfind(']')
            pos_str = parts[pos_start+1:pos_end]
            parts = parts[:pos_start].strip()
            
            try:
                pos_parts = pos_str.split(',')
                if len(pos_parts) >= 2:
                    position = (int(pos_parts[0].strip()), int(pos_parts[1].strip()))
            except ValueError:
                pass
        
        # Extract label if present
        if ':' in parts:
            idx = parts.index(':')
            node_type = parts[:idx].strip()
            label = parts[idx+1:].strip()
        else:
            node_type = parts.strip()
        
        return ASTDiffNode(
            node_type=node_type,
            label=label,
            position=position
        )
    
    def _parse_action(self, action_data: Dict[str, Any]) -> EditAction:
        """Parse a single action from GumTree JSON output"""
        action_type = EditType.from_string(action_data.get("action", "update"))
        
        # Parse the main node
        tree_str = action_data.get("tree", "")
        node = self._parse_tree_string(tree_str)
        
        # Parse parent if present
        parent = None
        parent_str = action_data.get("parent", "")
        if parent_str:
            parent = self._parse_tree_string(parent_str)
        
        # Extract position info
        at_position = action_data.get("at")
        
        return EditAction(
            action_type=action_type,
            node=node,
            parent=parent,
            at_position=at_position,
            raw_data=action_data
        )
    
    def diff(self, src_code: str, dst_code: str, wrap: bool = True) -> DiffResult:
        """
        Compute AST diff between source and destination code.
        
        Args:
            src_code: Source code (buggy version)
            dst_code: Destination code (fixed version)
            wrap: Whether to wrap code in dummy class (for method-level code)
            
        Returns:
            DiffResult containing all edit actions
        """
        if wrap:
            src_code = self._wrap_as_class(src_code)
            dst_code = self._wrap_as_class(dst_code)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            src_file = Path(tmpdir) / "Src.java"
            dst_file = Path(tmpdir) / "Dst.java"
            
            src_file.write_text(src_code, encoding='utf-8')
            dst_file.write_text(dst_code, encoding='utf-8')
            
            try:
                # Run GumTree diff command
                cmd = [
                    self.gumtree_path, "textdiff",
                    "-f", "json",
                    str(src_file), str(dst_file)
                ]
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    timeout=self.timeout,
                    shell=self.is_windows,
                    env=env
                )
                
                if result.returncode != 0:
                    print(f"GumTree error: {result.stderr}")
                    return DiffResult(actions=[])
                
                # Parse JSON output
                output = json.loads(result.stdout)
                
                # Parse actions
                actions = []
                for action_data in output.get("actions", []):
                    actions.append(self._parse_action(action_data))
                
                return DiffResult(
                    actions=actions,
                    raw_output=output
                )
                
            except subprocess.TimeoutExpired:
                print(f"GumTree timeout after {self.timeout}s")
                return DiffResult(actions=[])
            except json.JSONDecodeError as e:
                print(f"Failed to parse GumTree output: {e}")
                return DiffResult(actions=[])
            except Exception as e:
                print(f"Error running GumTree: {e}")
                return DiffResult(actions=[])
    
    def get_edit_script(self, src_code: str, dst_code: str, wrap: bool = True) -> str:
        """
        Get human-readable edit script.
        
        Returns:
            Text description of all edits
        """
        if wrap:
            src_code = self._wrap_as_class(src_code)
            dst_code = self._wrap_as_class(dst_code)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            src_file = Path(tmpdir) / "Src.java"
            dst_file = Path(tmpdir) / "Dst.java"
            
            src_file.write_text(src_code, encoding='utf-8')
            dst_file.write_text(dst_code, encoding='utf-8')
            
            try:
                cmd = [
                    self.gumtree_path, "textdiff",
                    "-f", "text",
                    str(src_file), str(dst_file)
                ]
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    timeout=self.timeout,
                    shell=self.is_windows,
                    env=env
                )
                
                return result.stdout if result.returncode == 0 else ""
                
            except Exception as e:
                print(f"Error: {e}")
                return ""
    
    def get_mappings(self, src_code: str, dst_code: str, wrap: bool = True) -> List[Dict]:
        """
        Get node mappings between source and destination ASTs.
        
        Returns:
            List of mappings showing which nodes correspond to each other
        """
        if wrap:
            src_code = self._wrap_as_class(src_code)
            dst_code = self._wrap_as_class(dst_code)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            src_file = Path(tmpdir) / "Src.java"
            dst_file = Path(tmpdir) / "Dst.java"
            
            src_file.write_text(src_code, encoding='utf-8')
            dst_file.write_text(dst_code, encoding='utf-8')
            
            try:
                cmd = [
                    self.gumtree_path, "textdiff",
                    "-f", "json",
                    str(src_file), str(dst_file)
                ]
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    timeout=self.timeout,
                    shell=self.is_windows,
                    env=env
                )
                
                if result.returncode == 0:
                    output = json.loads(result.stdout)
                    return output.get("matches", [])
                return []
                
            except Exception:
                return []


# Utility functions
def compute_diff(buggy_code: str, fixed_code: str, gumtree_path: str = None) -> DiffResult:
    """Quick function to compute diff between buggy and fixed code"""
    gt = GumTreeDiff(gumtree_path=gumtree_path)
    return gt.diff(buggy_code, fixed_code)


def get_edit_actions(buggy_code: str, fixed_code: str, gumtree_path: str = None) -> List[EditAction]:
    """Quick function to get list of edit actions"""
    result = compute_diff(buggy_code, fixed_code, gumtree_path)
    return result.actions


def format_diff_for_model(buggy_code: str, fixed_code: str, gumtree_path: str = None) -> str:
    """
    Format diff as text suitable for model input.
    
    Returns:
        String with each action on a line: [ACTION_TYPE] node_type: label
    """
    result = compute_diff(buggy_code, fixed_code, gumtree_path)
    return "\n".join(action.to_string() for action in result.actions)
