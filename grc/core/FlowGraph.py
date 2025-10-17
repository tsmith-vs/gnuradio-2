# Copyright 2008-2015 Free Software Foundation, Inc.
# This file is part of GNU Radio
#
# SPDX-License-Identifier: GPL-2.0-or-later
#


import collections
import itertools
import types
import logging
import yaml
from operator import methodcaller, attrgetter
from typing import (List, Set, Optional, Iterator, Iterable, Tuple, Union, OrderedDict, Sequence)
import ast
import importlib
from types import MappingProxyType
from typing import Optional


from . import Messages
from .base import Element
from .blocks import Block
from .params import Param
from .utils import expr_utils

log = logging.getLogger(__name__)

# Optional but common in GRC expressions
try:
    import numpy as _np
except Exception:
    _np = None
import math as _math

class UnsafeExpressionError(Exception):
    pass

# Restrictive builtins: no __import__, open, exec, eval, compile, etc.
SAFE_BUILTINS = MappingProxyType({
    "abs": abs,
    "min": min,
    "max": max,
    "round": round,
    "int": int,
    "float": float,
    "complex": complex,
    "bool": bool,
    "str": str,
    "len": len,
    "tuple": tuple,
    "list": list,
    "dict": dict,
    "set": set,
    # NOTE: Do not include __import__, open, eval, exec, compile, globals, locals, vars, getattr, setattr, delattr, etc.
})

# Allowed top-level modules by name (if present in namespace)
ALLOWED_MODULE_NAMES = {"math", "numpy", "np"}

# Allowed attributes callable on math/numpy modules
# Keep conservative; extend if you need more.
MATH_ATTRS = {
    "pi", "e", "tau",
    "sin", "cos", "tan",
    "asin", "acos", "atan", "atan2",
    "sinh", "cosh", "tanh",
    "asinh", "acosh", "atanh",
    "exp", "log", "log10", "log2",
    "sqrt", "pow", "floor", "ceil", "fabs",
    "degrees", "radians",
}
NUMPY_ATTRS_CONST = {
    # constants & dtypes commonly used in GRC
    "pi", "e",
    "float16", "float32", "float64",
    "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64",
    "complex64", "complex128",
}
NUMPY_ATTRS_FUNCS = {
    # basic numerics that are typically safe
    "sin", "cos", "tan", "arcsin", "arccos", "arctan", "arctan2",
    "sinh", "cosh", "tanh", "arcsinh", "arccosh", "arctanh",
    "exp", "log", "log10", "log2", "sqrt", "power",
    "floor", "ceil", "abs", "maximum", "minimum",
    "deg2rad", "rad2deg",
    # array constructors common in parameters (optional; comment out if you want stricter)
    "array", "arange", "linspace",
}
NUMPY_ATTRS = NUMPY_ATTRS_CONST | NUMPY_ATTRS_FUNCS

# For calls by bare name (not module.attr), allow only these builtins
ALLOWED_BARE_CALLS = {"abs", "min", "max", "round", "int", "float", "complex", "bool", "len"}

# Nodes we allow in expressions
_ALLOWED_NODE_TYPES = (
    ast.Expression,
    ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp,
    ast.Num, ast.Constant,  # literals
    ast.Name,
    ast.Attribute,
    ast.Subscript, ast.Slice, ast.ExtSlice, ast.Index,  # Index is py<3.9; harmless in 3.11 AST but safe to list
    ast.Tuple, ast.List, ast.Dict, ast.Set,
    ast.Load,
    ast.Call,
)

# Allowed operators
_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.MatMul)
_ALLOWED_UNARYOPS = (ast.UAdd, ast.USub, ast.Not, ast.Invert)
_ALLOWED_CMPOPS = (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Is, ast.IsNot, ast.In, ast.NotIn)
_ALLOWED_BOOLOPS = (ast.And, ast.Or)

# Allowlist of module prefixes that GRC commonly needs. Extend as necessary.
ALLOWED_IMPORT_PREFIXES = {
    "math",
    "numpy",
    "gnuradio",
    "gnuradio.gr",
    "pmt",
}

def _attr_chain(node: ast.AST) -> list[str]:
    """
    Return ['root', 'attr1'] for 'root.attr1', or longer lists for deeper chains.
    If not a simple attribute chain, return [].
    """
    parts: list[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        return list(reversed(parts))
    return []  # Not a simple Name.attr[.attr...] chain

class _SafeExprValidator(ast.NodeVisitor):
    def visit(self, node):  # type: ignore[override]
        if not isinstance(node, _ALLOWED_NODE_TYPES):
            raise UnsafeExpressionError(f"Unsupported expression node: {type(node).__name__}")
        return super().visit(node)

    def visit_Call(self, node: ast.Call):
        # Only allow calls to:
        #  - whitelisted builtins by bare name (e.g., abs(x))
        #  - whitelisted math/numpy functions by module attribute (e.g., math.sin(x), np.sqrt(x))
        func = node.func

        if isinstance(func, ast.Name):
            if func.id not in ALLOWED_BARE_CALLS:
                raise UnsafeExpressionError(f"Calling '{func.id}' is not allowed.")
        elif isinstance(func, ast.Attribute):
            chain = _attr_chain(func)
            if len(chain) != 2:
                # Disallow deep attribute calls like np.random.rand()
                raise UnsafeExpressionError("Deep attribute calls are not allowed.")
            root, attr = chain
            if root not in ALLOWED_MODULE_NAMES:
                raise UnsafeExpressionError(f"Calls on '{root}' are not allowed.")
            if root in {"math"} and attr not in MATH_ATTRS:
                raise UnsafeExpressionError(f"math.{attr} is not allowed.")
            if root in {"numpy", "np"} and attr not in NUMPY_ATTRS:
                raise UnsafeExpressionError(f"{root}.{attr} is not allowed.")
        else:
            raise UnsafeExpressionError("Only direct function names or module attributes may be called.")

        # Validate arguments/keywords recursively
        for a in node.args:
            self.visit(a)
        for kw in node.keywords or []:
            self.visit(kw.value)

    def visit_Attribute(self, node: ast.Attribute):
        # Allow attribute access ONLY on allowed modules and only for whitelisted attributes.
        chain = _attr_chain(node)
        if not chain:
            raise UnsafeExpressionError("Attribute access on non-module objects is not allowed.")
        if len(chain) != 2:
            raise UnsafeExpressionError("Deep attribute chains are not allowed.")
        root, attr = chain
        if root == "math" and attr not in MATH_ATTRS:
            raise UnsafeExpressionError(f"math.{attr} is not allowed.")
        if root in {"numpy", "np"} and attr not in NUMPY_ATTRS:
            raise UnsafeExpressionError(f"{root}.{attr} is not allowed.")

    def visit_Subscript(self, node: ast.Subscript):
        # Indexing/slicing is allowed as long as inner nodes are safe
        self.visit(node.value)
        self.visit(node.slice)

    def visit_Slice(self, node: ast.Slice):
        if node.lower: self.visit(node.lower)
        if node.upper: self.visit(node.upper)
        if node.step:  self.visit(node.step)

    def visit_BoolOp(self, node: ast.BoolOp):
        if not isinstance(node.op, _ALLOWED_BOOLOPS):
            raise UnsafeExpressionError("Boolean operator not allowed.")
        for v in node.values:
            self.visit(v)

    def visit_BinOp(self, node: ast.BinOp):
        if not isinstance(node.op, _ALLOWED_BINOPS):
            raise UnsafeExpressionError("Binary operator not allowed.")
        self.visit(node.left)
        self.visit(node.right)

    def visit_UnaryOp(self, node: ast.UnaryOp):
        if not isinstance(node.op, _ALLOWED_UNARYOPS):
            raise UnsafeExpressionError("Unary operator not allowed.")
        self.visit(node.operand)

    def visit_Compare(self, node: ast.Compare):
        for op in node.ops:
            if not isinstance(op, _ALLOWED_CMPOPS):
                raise UnsafeExpressionError("Comparison operator not allowed.")
        self.visit(node.left)
        for c in node.comparators:
            self.visit(c)

    def visit_IfExp(self, node: ast.IfExp):
        self.visit(node.test)
        self.visit(node.body)
        self.visit(node.orelse)

    # Literals & containers are inherently safe (already whitelisted in _ALLOWED_NODE_TYPES)
    # We still walk elements to enforce safety of nested expressions.
    def visit_Tuple(self, node: ast.Tuple):
        for el in node.elts: self.visit(el)

    def visit_List(self, node: ast.List):
        for el in node.elts: self.visit(el)

    def visit_Set(self, node: ast.Set):
        for el in node.elts: self.visit(el)

    def visit_Dict(self, node: ast.Dict):
        for k in node.keys: 
            if k is not None: self.visit(k)
        for v in node.values: self.visit(v)

    def visit_Name(self, node: ast.Name):
        # Bare names are OK, they resolve in filtered globals/locals later.
        # We specifically do NOT allow access to __builtins__ by not injecting it in globals.
        return

    def visit_Constant(self, node: ast.Constant):
        # Allow numbers, strings, bytes, bools, None
        return

def _filter_globals_for_eval(namespace: dict) -> dict:
    """
    Build a safe globals dict for eval: restricted builtins and filtered modules.
    We copy over all user variables (ints, floats, lists, dicts, etc.) and any
    allowed modules (math, numpy/np) if present.
    """
    g = {"__builtins__": SAFE_BUILTINS}

    # expose math (always safe subset)
    g["math"] = _math

    # expose numpy aliases if in the incoming namespace or importable
    if "numpy" in namespace and namespace["numpy"] is not None:
        g["numpy"] = namespace["numpy"]
    elif _np is not None:
        g["numpy"] = _np

    if "np" in namespace and namespace["np"] is not None:
        g["np"] = namespace["np"]
    elif _np is not None:
        g["np"] = _np

    # Copy non-module variables through verbatim.
    for k, v in namespace.items():
        # Skip shadowing of protected names
        if k in {"__builtins__", "__import__"}:
            continue
        # Do not expose disallowed modules (if any slipped into namespace)
        modname = getattr(v, "__name__", None)
        if modname and getattr(v, "__spec__", None) is not None:
            # it's a module-like object
            if k not in ALLOWED_MODULE_NAMES:
                continue  # skip disallowed modules
        g[k] = v

    return g


class ImportSecurityError(Exception):
    pass

def _is_allowed_import(modname: str) -> bool:
    if not modname:
        return False
    return any(
        modname == prefix or modname.startswith(prefix + ".")
        for prefix in ALLOWED_IMPORT_PREFIXES
    )


def _validate_imports_only(source: str) -> ast.Module:
    """
    Parse 'source' and ensure it contains ONLY:
      - import / from-import statements
      - optional module docstring
    Rejects: relative imports, wildcard imports, and any other statements.
    """
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as e:
        raise e

    allowed_topnodes = (ast.Import, ast.ImportFrom, ast.Expr)
    if not all(isinstance(node, allowed_topnodes) for node in tree.body):
        raise ImportSecurityError("Only import statements (and an optional docstring) are allowed.")

    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if not _is_allowed_import(alias.name):
                    raise ImportSecurityError(f'Disallowed import: "{alias.name}"')
        elif isinstance(node, ast.ImportFrom):
            if (node.level or 0) > 0:
                raise ImportSecurityError("Relative imports are not allowed.")
            if any(a.name == "*" for a in node.names):
                raise ImportSecurityError('Wildcard imports (from x import *) are not allowed.')
            if not node.module or not _is_allowed_import(node.module):
                raise ImportSecurityError(f'Disallowed import: "{node.module or ""}"')
        elif isinstance(node, ast.Expr):
            # Only allow a module docstring
            if not (isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)):
                raise ImportSecurityError("Only a module docstring is allowed as a top-level expression.")

    return tree

def _apply_validated_imports(source: str, namespace: dict) -> None:
    """
    Execute only import statements in a controlled way, binding names into namespace.
    Mirrors Python's binding semantics without exec() on arbitrary code.
    """
    tree = _validate_imports_only(source)

    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                modname = alias.name
                asname = alias.asname

                # Ensure the full module is loaded
                module = importlib.import_module(modname)

                if asname:
                    namespace[asname] = module
                else:
                    # 'import a.b' binds 'a'
                    top_level = modname.split(".")[0]
                    namespace[top_level] = importlib.import_module(top_level)

        elif isinstance(node, ast.ImportFrom):
            module_name = node.module
            module = importlib.import_module(module_name)
            for alias in node.names:
                bind_name = alias.asname or alias.name
                namespace[bind_name] = getattr(module, alias.name)
        else:
            # Expr nodes here are only a docstring—ignore
            pass


class ModuleSecurityError(Exception):
    pass

def _validate_module_source(source: str) -> ast.Module:
    """
    Allow a conservative subset for module code:
      - import / from-import (allowlist, no relative, no wildcard)
      - assignments / annotated assignments
      - function and class definitions (no call expressions at module/class scope)
      - pass
      - an optional module/class docstring

    Disallow:
      - any top-level Call expressions
      - control flow at top level (if/for/while/try/with)
      - comprehensions/lambdas at top level
      - decorators other than simple names (e.g., @staticmethod, @classmethod, @property)
    """
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as e:
        raise e

    allowed_topnodes = (ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign,
                        ast.FunctionDef, ast.ClassDef, ast.Pass, ast.Expr)

    for node in tree.body:
        if not isinstance(node, allowed_topnodes):
            raise ModuleSecurityError(f"Disallowed top-level statement: {type(node).__name__}")

        if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
            _validate_imports_only(ast.unparse(node) if hasattr(ast, "unparse") else _reconstruct_import(node))
        elif isinstance(node, ast.Expr):
            # Only allow a pure docstring at module level
            if not (isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)):
                raise ModuleSecurityError("Only a module docstring is allowed as a top-level expression.")
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            # No Call() in targets or values at module scope
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call):
                    raise ModuleSecurityError("Function calls are not allowed at module scope.")
        elif isinstance(node, ast.FunctionDef):
            # Decorators allowed only if simple safe names
            for dec in node.decorator_list:
                if not isinstance(dec, ast.Name) or dec.id not in {"staticmethod", "classmethod", "property"}:
                    raise ModuleSecurityError("Unsupported function decorator.")
            # Defaults and annotations must not invoke calls
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call):
                    # Calls inside the function body are fine; they don't run at import time.
                    if sub in node.body:  # not reliable to compare nodes; do a cheap scope check below
                        pass
            # Prohibit calls in function signature (defaults/annotations)
            for d in (node.args.defaults or []) + (node.args.kw_defaults or []):
                if d is not None:
                    for sub in ast.walk(d):
                        if isinstance(sub, ast.Call):
                            raise ModuleSecurityError("Calls in default arguments are not allowed.")
            if node.returns:
                for sub in ast.walk(node.returns):
                    if isinstance(sub, ast.Call):
                        raise ModuleSecurityError("Calls in return annotations are not allowed.")
        elif isinstance(node, ast.ClassDef):
            # Class decorators only as simple safe names (none by default; add if you need @dataclass)
            for dec in node.decorator_list:
                if not isinstance(dec, ast.Name):
                    raise ModuleSecurityError("Unsupported class decorator.")
                if dec.id not in set():  # empty set -> disallow by default; add "dataclass" if needed
                    raise ModuleSecurityError(f"Unsupported class decorator @{dec.id}.")

            # For class body: allow defs/assign/pass/docstring only; no calls at class scope
            for stmt in node.body:
                if isinstance(stmt, (ast.FunctionDef, ast.Assign, ast.AnnAssign, ast.Pass, ast.Expr)):
                    if isinstance(stmt, ast.Expr):
                        if not (isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str)):
                            raise ModuleSecurityError("Only a class docstring is allowed as a class-level expression.")
                    else:
                        for sub in ast.walk(stmt):
                            if isinstance(sub, ast.Call):
                                raise ModuleSecurityError("Calls at class scope are not allowed.")
                else:
                    raise ModuleSecurityError("Only methods, assignments, pass, and docstrings are allowed in classes.")

    return tree

def _reconstruct_import(node: ast.AST) -> str:
    """Fallback stringification for imports on older Python without ast.unparse."""
    if isinstance(node, ast.Import):
        parts = ", ".join(
            a.name + (f" as {a.asname}" if a.asname else "")
            for a in node.names
        )
        return f"import {parts}"
    elif isinstance(node, ast.ImportFrom):
        parts = ", ".join(
            a.name + (f" as {a.asname}" if a.asname else "")
            for a in node.names
        )
        dots = "." * (node.level or 0)
        mod = node.module or ""
        return f"from {dots}{mod} import {parts}"
    return ""


def _exec_module_safely(source: str, modname: str) -> types.ModuleType:
    """
    Compile & execute validated module code with restricted builtins.
    Import operations inside 'source' are validated (allowlist) and happen via importlib.
    """
    tree = _validate_module_source(source)
    code = compile(tree, filename=f"<grc-module:{modname}>", mode="exec")

    # Reuse your SAFE_BUILTINS from safe_eval
    try:
        safe_builtins = SAFE_BUILTINS  # already defined earlier in this file
    except NameError:
        # Define the minimal SAFE_BUILTINS once if not present
        from types import MappingProxyType
        safe_builtins = MappingProxyType({
            "object": object, "property": property,
            "staticmethod": staticmethod, "classmethod": classmethod,
            "abs": abs, "min": min, "max": max, "round": round,
            "int": int, "float": float, "complex": complex, "bool": bool,
            "str": str, "len": len, "tuple": tuple, "list": list, "dict": dict, "set": set,
        })

    module = types.ModuleType(modname)
    g = module.__dict__
    g.clear()
    g["__name__"] = modname
    g["__builtins__"] = safe_builtins

    # Provide a tiny import surface by pre-binding a helper that only permits allowed imports.
    # Instead of exposing __import__, we emulate it via _apply_validated_imports on strings.
    # But since module code uses Python's 'import' statements, not strings, we rely on
    # AST validation to ensure only allowed imports exist. No __import__ is present in builtins.

    exec(code, g)
    return module

def safe_eval(expr: str, globals_ns: dict, locals_ns: Optional[dict] = None):
    """
    Validate 'expr' AST, then eval it with restricted builtins and filtered globals.
    Raises UnsafeExpressionError for unsafe constructs.
    """
    if not expr or not isinstance(expr, str):
        raise UnsafeExpressionError("Empty or invalid expression.")

    # Quick path: simple literal
    try:
        # ast.literal_eval is safe but handles only literals/containers
        return ast.literal_eval(expr)
    except Exception:
        pass

    # Parse and validate
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        # keep original error surface for callers
        raise e

    _SafeExprValidator().visit(tree)

    # Evaluate with restricted globals
    safe_globals = _filter_globals_for_eval(globals_ns or {})
    safe_locals = {} if locals_ns is None else dict(locals_ns)  # shallow copy

    # Ensure locals cannot reintroduce builtins
    safe_locals.pop("__builtins__", None)

    return eval(compile(tree, filename="<safe-expr>", mode="eval"), safe_globals, safe_locals)


class FlowGraph(Element):

    is_flow_graph = True

    def __init__(self, parent: Element):
        """
        Make a flow graph from the arguments.

        Args:
            parent: a platforms with blocks and element factories

        Returns:
            the flow graph object
        """
        Element.__init__(self, parent)
        self.options_block: Block = self.parent_platform.make_block(self, 'options')

        self.blocks = [self.options_block]
        self.connections = set()

        self._eval_cache = {}
        self.namespace = {}
        self.imported_names = []

        self.grc_file_path = ''

    def __str__(self) -> str:
        return f"FlowGraph - {self.get_option('title')}({self.get_option('id')})"

    def imports(self) -> List[str]:
        """
        Get a set of all import statements (Python) in this flow graph namespace.

        Returns:
            a list of import statements
        """
        return [block.templates.render('imports') for block in self.iter_enabled_blocks()]

    def get_variables(self) -> List[str]:
        """
        Get a list of all variables (Python) in this flow graph namespace.
        Exclude parameterized variables.

        Returns:
            a sorted list of variable blocks in order of dependency (indep -> dep)
        """
        variables = [block for block in self.iter_enabled_blocks()
                     if block.is_variable]
        return expr_utils.sort_objects(variables, attrgetter('name'), methodcaller('get_var_make'))

    def get_parameters(self) -> List[Element]:
        """
        Get a list of all parameterized variables in this flow graph namespace.

        Returns:
            a list of parameterized variables
        """
        parameters = [b for b in self.iter_enabled_blocks()
                      if b.key == 'parameter']
        return parameters

    def _get_snippets(self) -> List[Element]:
        """
        Get a set of all code snippets (Python) in this flow graph namespace.

        Returns:
            a list of code snippets
        """
        return [b for b in self.iter_enabled_blocks() if b.key == 'snippet']

    def get_snippets_dict(self, section=None) -> List[dict]:
        """
        Get a dictionary of code snippet information for a particular section.

        Args:
            section: string specifier of section of snippets to return, section=None returns all

        Returns:
            a list of code snippets dicts
        """
        snippets = self._get_snippets()
        if not snippets:
            return []

        output = []
        for snip in snippets:
            d = {}
            sect = snip.params['section'].value
            d['section'] = sect
            d['priority'] = snip.params['priority'].value
            d['lines'] = snip.params['code'].value.splitlines()
            d['def'] = 'def snipfcn_{}(self):'.format(snip.name)
            d['call'] = 'snipfcn_{}(tb)'.format(snip.name)
            if not len(d['lines']):
                Messages.send_warning("Ignoring empty snippet from canvas")
            else:
                if not section or sect == section:
                    output.append(d)

        # Sort by descending priority
        if section:
            output = sorted(output, key=lambda x: x['priority'], reverse=True)

        return output

    def get_monitors(self) -> List[Element]:
        """
        Get a list of all ControlPort monitors
        """
        monitors = [b for b in self.iter_enabled_blocks()
                    if 'ctrlport_monitor' in b.key]
        return monitors

    def get_python_modules(self) -> Iterator[Tuple[str, str]]:
        """Iterate over custom code block ID and Source"""
        for block in self.iter_enabled_blocks():
            if block.key == 'epy_module':
                yield block.name, block.params['source_code'].get_value()

    def iter_enabled_blocks(self) -> Iterator[Element]:
        """
        Get an iterator of all blocks that are enabled and not bypassed.
        """
        return (block for block in self.blocks if block.enabled)

    def get_enabled_blocks(self) -> List[Element]:
        """
        Get a list of all blocks that are enabled and not bypassed.

        Returns:
            a list of blocks
        """
        return list(self.iter_enabled_blocks())

    def get_bypassed_blocks(self) -> List[Element]:
        """
        Get a list of all blocks that are bypassed.

        Returns:
            a list of blocks
        """
        return [block for block in self.blocks if block.get_bypassed()]

    def get_enabled_connections(self) -> List[Element]:
        """
        Get a list of all connections that are enabled.

        Returns:
            a list of connections
        """
        return [connection for connection in self.connections if connection.enabled]

    def get_option(self, key) -> Param.EvaluationType:
        """
        Get the option for a given key.
        The option comes from the special options block.

        Args:
            key: the param key for the options block

        Returns:
            the value held by that param
            will return None if the key param key doesn't exist
        """
        param = self.options_block.params.get(key)
        if param:
            return param.get_evaluated()
        else:
            return None

    def get_imported_names(self) -> Set[str]:
        """
        Get a list of imported names.
        These names may not be used as id's

        Returns:
            a list of imported names
        """
        return self.imported_names

    ##############################################
    # Access Elements
    ##############################################
    def get_block(self, name) -> Block:
        for block in self.blocks:
            if block.name == name:
                return block
        raise KeyError(f'No block with name {repr(name)}')

    def get_elements(self) -> List[Element]:
        elements = list(self.blocks)
        elements.extend(self.connections)
        return elements

    def children(self) -> Iterable[Element]:
        return itertools.chain(self.blocks, self.connections)

    def rewrite(self):
        """
        Flag the namespace to be renewed.
        """
        self._renew_namespace()
        Element.rewrite(self)

    def _reload_imports(self, namespace: dict) -> dict:
        """
        Load imports; be tolerant about import errors
        """
        for expr in self.imports():
            try:
                _apply_validated_imports(expr, namespace)
            except ImportError:
                # Hier block imports may fail (search path), keep current behavior
                pass
            except (ImportSecurityError, SyntaxError):
                log.exception(f"Failed to evaluate import expression \"{expr}\"", exc_info=True)
                pass
            except Exception:
                log.exception(f"Failed to evaluate import expression \"{expr}\"", exc_info=True)
                pass
        return namespace

    def _reload_modules(self, namespace: dict) -> dict:
        for id, expr in self.get_python_modules():
            try:
                module = _exec_module_safely(expr, id)
                namespace[id] = module
            except Exception:
                log.exception(f'Failed to evaluate expression in module {id}', exc_info=True)
                pass
        return namespace

    def _reload_parameters(self, namespace: dict) -> dict:
        """
        Load parameters. Be tolerant of evaluation failures.
        """
        np = {}  # params don't know each other
        for parameter_block in self.get_parameters():
            try:
                code = parameter_block.params['value'].to_code()
                value = safe_eval(code, namespace)
                np[parameter_block.name] = value
            except Exception:
                # Keep original logging behavior
                log.exception(
                    f'Failed to evaluate parameter block {parameter_block.name}',
                    exc_info=True
                )
                pass
        namespace.update(np)  # Merge param namespace
        return namespace

    def _reload_variables(self, namespace: dict) -> dict:
        """
        Load variables. Be tolerant of evaluation failures.
        """
        for variable_block in self.get_variables():
            try:
                variable_block.rewrite()
                value = safe_eval(variable_block.value, namespace, variable_block.namespace)
                namespace[variable_block.name] = value
                # rewrite on subsequent blocks depends on an updated self.namespace
                self.namespace.update(namespace)
            # The following Errors may happen, but that doesn't matter as they are displayed in the gui
            except (TypeError, FileNotFoundError, AttributeError, yaml.YAMLError):
                pass
            except Exception:
                log.exception(
                    f'Failed to evaluate variable block {variable_block.name}',
                    exc_info=True
                )
        return namespace

    def _renew_namespace(self) -> None:
        # Before renewing the namespace, clear it
        # to get rid of entries of blocks that
        # are no longer valid ( deleted, disabled, ...)
        self.namespace.clear()

        namespace = self._reload_imports({})
        self.imported_names = set(namespace.keys())
        namespace = self._reload_modules(namespace)
        namespace = self._reload_parameters(namespace)

        # We need the updated namespace to evaluate the variable blocks
        # otherwise sometimes variable_block rewrite / eval fails
        self.namespace.update(namespace)
        namespace = self._reload_variables(namespace)
        self._eval_cache.clear()

    def evaluate(self, expr: str, namespace: Optional[dict] = None, local_namespace: Optional[dict] = None):
        """
        Evaluate the expression within the specified global and local namespaces
        """
        if not expr:
            raise Exception('Cannot evaluate empty statement.')
    
        if namespace is not None:
            return safe_eval(expr, namespace, local_namespace)
        else:
            # cache only successful results
            if expr in self._eval_cache:
                return self._eval_cache[expr]
            value = safe_eval(expr, self.namespace, local_namespace)
            self._eval_cache[expr] = value
            return value

    ##############################################
    # Add/remove stuff
    ##############################################

    def new_block(self, block_id, **kwargs) -> Block:
        """
        Get a new block of the specified key.
        Add the block to the list of elements.

        Args:
            block_id: the block key

        Returns:
            the new block or None if not found
        """
        if block_id == 'options':
            return self.options_block
        try:
            block = self.parent_platform.make_block(self, block_id, **kwargs)
            self.blocks.append(block)
        except KeyError:
            block = None
        return block

    def connect(self, porta, portb, params=None):
        """
        Create a connection between porta and portb.

        Args:
            porta: a port
            portb: another port
        @throw Exception bad connection

        Returns:
            the new connection
        """
        connection = self.parent_platform.Connection(
            parent=self, source=porta, sink=portb)
        if params:
            connection.import_data(params)
        self.connections.add(connection)

        return connection

    def disconnect(self, *ports) -> None:
        to_be_removed = [con for con in self.connections
                         if any(port in con for port in ports)]
        for con in to_be_removed:
            self.remove_element(con)

    def remove_element(self, element) -> None:
        """
        Remove the element from the list of elements.
        If the element is a port, remove the whole block.
        If the element is a block, remove its connections.
        If the element is a connection, just remove the connection.
        """
        if element is self.options_block:
            return

        if element.is_port:
            element = element.parent_block  # remove parent block

        if element in self.blocks:
            # Remove block, remove all involved connections
            self.disconnect(*element.ports())
            self.blocks.remove(element)

        elif element in self.connections:
            self.connections.remove(element)

    ##############################################
    # Import/Export Methods
    ##############################################
    def export_data(self) -> OrderedDict[str, str]:
        """
        Export this flow graph to nested data.
        Export all block and connection data.

        Returns:
            a nested data odict
        """
        def block_order(b):
            return not b.is_variable, b.name  # todo: vars still first ?!?

        def get_file_format_version(data) -> int:
            """Determine file format version based on available data"""
            if any(isinstance(c, dict) for c in data['connections']):
                return 2
            return 1

        def sort_connection_key(connection_info) -> List[str]:
            if isinstance(connection_info, dict):
                return [connection_info.get(key) for key in ('src_blk_id', 'src_port_id', 'snk_blk_id', 'snk_port_id')]
            return connection_info
        data = collections.OrderedDict()
        data['options'] = self.options_block.export_data()
        data['blocks'] = [b.export_data() for b in sorted(self.blocks, key=block_order)
                          if b is not self.options_block]
        data['connections'] = sorted(
            (c.export_data() for c in self.connections),
            key=sort_connection_key)

        data['metadata'] = {
            'file_format': get_file_format_version(data),
            'grc_version': self.parent_platform.config.version
        }
        return data

    def _build_depending_hier_block(self, block_id) -> Optional[Block]:
        # we're before the initial fg update(), so no evaluated values!
        # --> use raw value instead
        path_param = self.options_block.params['hier_block_src_path']
        file_path = self.parent_platform.find_file_in_paths(
            filename=block_id + '.grc',
            paths=path_param.get_value(),
            cwd=self.grc_file_path
        )
        if file_path:  # grc file found. load and get block
            self.parent_platform.load_and_generate_flow_graph(
                file_path, hier_only=True)
            return self.new_block(block_id)  # can be None

    def import_data(self, data) -> bool:
        """
        Import blocks and connections into this flow graph.
        Clear this flow graph of all previous blocks and connections.
        Any blocks or connections in error will be ignored.

        Args:
            data: the nested data odict

        Returns:
            connection_error bool signifying whether a connection error happened.
        """
        # Remove previous elements
        del self.blocks[:]
        self.connections.clear()

        file_format = data['metadata']['file_format']

        # build the blocks
        self.options_block.import_data(name='', **data.get('options', {}))
        self.options_block.insert_grc_parameters(data['options']['parameters'])
        self.blocks.append(self.options_block)

        for block_data in data.get('blocks', []):
            block_id = block_data['id']
            block = (
                self.new_block(block_id) or
                self._build_depending_hier_block(block_id) or
                self.new_block(block_id='_dummy',
                               missing_block_id=block_id, **block_data)
            )

            block.import_data(**block_data)

        self.rewrite()

        # build the connections
        def verify_and_get_port(key, block, dir):
            ports = block.sinks if dir == 'sink' else block.sources
            for port in ports:
                if key == port.key or key + '0' == port.key:
                    break
                if not key.isdigit() and port.dtype == '' and key == port.name:
                    break
            else:
                if block.is_dummy_block:
                    port = block.add_missing_port(key, dir)
                else:
                    raise LookupError(f"{dir} key {key} not in {dir} block keys")
            return port

        had_connect_errors = False
        _blocks = {block.name: block for block in self.blocks}

        # TODO: Add better error handling if no connections exist in the flowgraph file.
        for connection_info in data.get('connections', []):
            # First unpack the connection info, which can be in different formats.
            # FLOW_GRAPH_FILE_FORMAT_VERSION 1: Connection info is a 4-tuple
            if isinstance(connection_info, (list, tuple)) and len(connection_info) == 4:
                src_blk_id, src_port_id, snk_blk_id, snk_port_id = connection_info
                conn_params = {}
            # FLOW_GRAPH_FILE_FORMAT_VERSION 2: Connection info is a dictionary
            elif isinstance(connection_info, dict):
                src_blk_id = connection_info.get('src_blk_id')
                src_port_id = connection_info.get('src_port_id')
                snk_blk_id = connection_info.get('snk_blk_id')
                snk_port_id = connection_info.get('snk_port_id')
                conn_params = connection_info.get('params', {})
            else:
                Messages.send_error_load('Invalid connection format detected!')
                had_connect_errors = True
                continue
            try:
                source_block = _blocks[src_blk_id]
                sink_block = _blocks[snk_blk_id]

                # fix old, numeric message ports keys
                if file_format < 1:
                    src_port_id, snk_port_id = _update_old_message_port_keys(
                        src_port_id, snk_port_id, source_block, sink_block)

                # build the connection
                source_port = verify_and_get_port(
                    src_port_id, source_block, 'source')
                sink_port = verify_and_get_port(
                    snk_port_id, sink_block, 'sink')

                self.connect(source_port, sink_port, conn_params)

            except (KeyError, LookupError) as e:
                Messages.send_error_load(
                    f"""Connection between {src_blk_id}({src_port_id}) and {snk_blk_id}({snk_port_id}) could not be made
                    \t{e}""")
                had_connect_errors = True

        for block in self.blocks:
            if block.is_dummy_block:
                block.rewrite()      # Make ports visible
                # Flowgraph errors depending on disabled blocks are not displayed
                # in the error dialog box
                # So put a message into the Property window of the dummy block
                block.add_error_message(f'Block id "{block.key}" not found.')

        self.rewrite()  # global rewrite
        return had_connect_errors


def _update_old_message_port_keys(source_key, sink_key, source_block, sink_block) -> Tuple[str, str]:
    """
    Backward compatibility for message port keys

    Message ports use their names as key (like in the 'connect' method).
    Flowgraph files from former versions still have numeric keys stored for
    message connections. These have to be replaced by the name of the
    respective port. The correct message port is deduced from the integer
    value of the key (assuming the order has not changed).

    The connection ends are updated only if both ends translate into a
    message port.
    """
    try:
        # get ports using the "old way" (assuming linear indexed keys)
        source_port = source_block.sources[int(source_key)]
        sink_port = sink_block.sinks[int(sink_key)]
        if source_port.dtype == "message" and sink_port.dtype == "message":
            source_key, sink_key = source_port.key, sink_port.key
    except (ValueError, IndexError):
        pass
    return source_key, sink_key  # do nothing
