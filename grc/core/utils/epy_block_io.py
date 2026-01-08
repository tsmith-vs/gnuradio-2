
import ast
import builtins as py_builtins
import importlib
import types
import inspect
import collections


TYPE_MAP = {
    'complex64': 'complex', 'complex': 'complex',
    'float32': 'float', 'float': 'float',
    'int32': 'int', 'uint32': 'int',
    'int16': 'short', 'uint16': 'short',
    'int8': 'byte', 'uint8': 'byte',
}

BlockIO = collections.namedtuple(
    'BlockIO', 'name cls params sinks sources doc callbacks')

class SecurityError(Exception):
    """Raised when user-supplied source violates safety rules."""


# Minimal builtins typically needed for class definitions & simple code.
# Strictly avoid __import__, exec, eval, open, compile, etc.
SAFE_BUILTINS = {
    "object": py_builtins.object,
    "property": py_builtins.property,
    "staticmethod": py_builtins.staticmethod,
    "classmethod": py_builtins.classmethod,
    "len": py_builtins.len,
    "range": py_builtins.range,
    "enumerate": py_builtins.enumerate,
    "zip": py_builtins.zip,
    "min": py_builtins.min,
    "max": py_builtins.max,
    "sum": py_builtins.sum,
    "abs": py_builtins.abs,
    "int": py_builtins.int,
    "float": py_builtins.float,
    "complex": py_builtins.complex,
    "bool": py_builtins.bool,
    "str": py_builtins.str,
    "bytes": py_builtins.bytes,
    "list": py_builtins.list,
    "tuple": py_builtins.tuple,
    "dict": py_builtins.dict,
    "set": py_builtins.set,
}

# Allowlisted modules that are common/expected in EPY blocks.
# This can be exetended via configuration if needed.
ALLOWED_IMPORT_PREFIXES = {
    "numpy",
    "gnuradio",
    "gnuradio.gr",
    "pmt",
}


def _is_allowed_import(modname: str) -> bool:
    if not modname:
        return False
    return any(
        modname == prefix or modname.startswith(prefix + ".")
        for prefix in ALLOWED_IMPORT_PREFIXES
    )


def _validate_epy_module_ast(source: str) -> ast.Module:
    """
    Parse and validate that the source contains only:
      - import / from-import (no relative, no wildcard, allowlist enforced)
      - class definitions (limited decorators)
      - optional module docstring

    We reject:
      - any other top-level statements (calls, with, try, for, etc.)
      - relative or wildcard imports
      - disallowed imports
      - suspicious decorators at class level
    """
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as e:
        raise SyntaxError(f"Syntax error in EPY block: {e}") from e

    allowed_topnodes = (ast.Import, ast.ImportFrom, ast.ClassDef, ast.Expr)
    if not all(isinstance(node, allowed_topnodes) for node in tree.body):
        raise SecurityError("Only imports, class definitions, and docstrings are allowed at module level.")

    # Validate imports
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                modname = alias.name
                if not _is_allowed_import(modname):
                    raise SecurityError(f'Disallowed import: "{modname}"')
        elif isinstance(node, ast.ImportFrom):
            if (node.level or 0) > 0:
                raise SecurityError("Relative imports are not allowed.")
            if any(a.name == "*" for a in node.names):
                raise SecurityError('Wildcard imports (from x import *) are not allowed.')
            if not node.module or not _is_allowed_import(node.module):
                raise SecurityError(f'Disallowed import: "{node.module or ""}"')
        elif isinstance(node, ast.Expr):
            # Only allow a docstring expr at module level
            if not isinstance(node.value, ast.Constant) or not isinstance(node.value.value, str):
                raise SecurityError("Only a module docstring is allowed as a top-level expression.")
        elif isinstance(node, ast.ClassDef):
            # Validate class decorators (only common safe ones)
            for dec in node.decorator_list:
                if not isinstance(dec, ast.Name):
                    raise SecurityError("Unsupported class decorator.")
                if dec.id not in {"dataclass"}:  # Extend if you want to allow @dataclass
                    # We allow no decorators by default; comment out the next line if @dataclass is not desired.
                    raise SecurityError(f"Unsupported class decorator @{dec.id}.")

            # Limit class-level statements: functions, assignments, docstrings, pass
            for stmt in node.body:
                if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.AnnAssign, ast.Assign, ast.Pass)):
                    # Disallow Calls in default values at class scope to reduce side effects
                    for sub in ast.walk(stmt):
                        if isinstance(sub, ast.Call):
                            raise SecurityError("Function calls at class scope are not allowed.")
                elif isinstance(stmt, ast.Expr):
                    # docstring allowed
                    if not (isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str)):
                        raise SecurityError("Only a docstring is allowed as a class-level expression.")
                else:
                    raise SecurityError("Only methods, assignments, pass, and docstrings are allowed inside classes.")

    return tree


def _load_epy_module_safely(source: str) -> types.ModuleType:
    """
    Compile & execute validated AST in an isolated module dict with restricted builtins.
    Pre-imports a small set of allowed modules and injects them into the globals.
    """
    tree = _validate_epy_module_ast(source)
    code = compile(tree, filename="<grc-epy>", mode="exec")

    # Build isolated globals with restricted builtins
    g: dict = {"__name__": "<grc-epy>", "__builtins__": SAFE_BUILTINS}

    # Pre-import the allowlisted modules that EPY blocks typically rely on
    # and provide common aliases (e.g., numpy as np)
    try:
        gr = importlib.import_module("gnuradio.gr")
        g["gr"] = gr
    except Exception:
        # The caller (extract) separately handles missing GNU Radio
        pass

    try:
        np = importlib.import_module("numpy")
        g["numpy"] = np
        g["np"] = np
    except Exception:
        pass

    try:
        pmt = importlib.import_module("pmt")
        g["pmt"] = pmt
    except Exception:
        pass

    # Execute the validated code object in the isolated namespace
    # NOTE: executing code is unavoidable to materialize classes.
    exec(code, g)  # safer because code is AST-validated & builtins are restricted

    # Wrap as a module-like object for consistency with inspect workflows
    mod = types.ModuleType("<grc-epy>")
    mod.__dict__.update(g)
    return mod


def _find_block_class_safely(source_code: str, cls) -> type:
    """
    Load user code in a restricted environment and return the first class
    that is a subclass of `cls`. Raises ValueError if not found or unsafe.
    """
    mod = _load_epy_module_safely(source_code)

    for var in mod.__dict__.values():
        if inspect.isclass(var) and issubclass(var, cls):
            return var
    raise ValueError("No python block class found in code")

def _ports(sigs, msgs):
    ports = list()
    for i, dtype in enumerate(sigs):
        port_type = TYPE_MAP.get(dtype.base.name, None)
        if not port_type:
            raise ValueError("Can't map {0!r} to GRC port type".format(dtype))
        vlen = dtype.shape[0] if len(dtype.shape) > 0 else 1
        ports.append((str(i), port_type, vlen))
    for msg_key in msgs:
        if msg_key == 'system':
            continue
        ports.append((msg_key, 'message', 1))
    return ports


def _find_block_class(source_code, cls):
    try:
        return _find_block_class_safely(source_code, cls)
    except (SyntaxError, SecurityError) as e:
        raise ValueError("Can't interpret source code: " + str(e)) from e
    except Exception as e:
        # Preserve previous error contract
        raise ValueError("Can't interpret source code: " + str(e)) from e


def extract(cls):
    try:
        from gnuradio import gr
        import pmt
    except ImportError:
        raise EnvironmentError("Can't import GNU Radio")

    if not inspect.isclass(cls):
        cls = _find_block_class(cls, gr.gateway.gateway_block)

    spec = inspect.getfullargspec(cls.__init__)
    init_args = spec.args[1:]
    defaults = [repr(arg) for arg in (spec.defaults or ())]
    doc = cls.__doc__ or cls.__init__.__doc__ or ''
    cls_name = cls.__name__

    if len(defaults) + 1 != len(spec.args):
        raise ValueError("Need all __init__ arguments to have default values")

    try:
        instance = cls()
    except Exception as e:
        raise RuntimeError("Can't create an instance of your block: " + str(e))

    name = instance.name()

    params = list(zip(init_args, defaults))

    def settable(attr):
        try:
            # check for a property with setter
            return callable(getattr(cls, attr).fset)
        except AttributeError:
            return attr in instance.__dict__  # not dir() - only the instance attribs

    callbacks = [attr for attr in dir(
        instance) if attr in init_args and settable(attr)]

    sinks = _ports(instance.in_sig(),
                   pmt.to_python(instance.message_ports_in()))
    sources = _ports(instance.out_sig(),
                     pmt.to_python(instance.message_ports_out()))

    return BlockIO(name, cls_name, params, sinks, sources, doc, callbacks)


if __name__ == '__main__':
    blk_code = """
import numpy as np
from gnuradio import gr
import pmt

class blk(gr.sync_block):
    def __init__(self, param1=None, param2=None, param3=None):
        "Test Docu"
        gr.sync_block.__init__(
            self,
            name='Embedded Python Block',
            in_sig = (np.float32,),
            out_sig = (np.float32,np.complex64,),
        )
        self.message_port_register_in(pmt.intern('msg_in'))
        self.message_port_register_out(pmt.intern('msg_out'))
        self.param1 = param1
        self._param2 = param2
        self._param3 = param3

    @property
    def param2(self):
        return self._param2

    @property
    def param3(self):
        return self._param3

    @param3.setter
    def param3(self, value):
        self._param3 = value

    def work(self, inputs_items, output_items):
        return 10
    """
    from pprint import pprint
    pprint(dict(extract(blk_code)._asdict()))
