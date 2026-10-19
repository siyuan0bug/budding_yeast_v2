"""Source-derived *potential dependency* graphs, not fixed signed causal graphs.

Only the ODE, its reachable scalar helpers, event guards, and reset dispatch in
mutant_aware.py are inspected. The source module is never imported/executed.
No default/real-mutant parameter table or trajectory is used. All matrix views
are derived from immutable equation/event records, with A[target, source].

This deliberately bounded AST interpreter fails on unsupported syntax. It is
not a general Python analyzer, a sign prover, or a saltation/sensitivity solver.
SymPy normalizes scalar expressions; numerical Jacobians are validation only.
Existing model loaders/checkpoints are intentionally not changed in phase 1.
"""
from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
import sympy as sp
from sympy.logic.boolalg import Boolean


DEFAULT_SOURCE = Path(__file__).resolve().with_name("mutant_aware.py")
SCHEMA_VERSION = 1
VIEWS = ("continuous", "event_trigger", "reset_value", "event", "hybrid")


class GraphExtractionError(ValueError):
    """The source no longer satisfies the supported, explicit extraction contract."""


@dataclass(frozen=True)
class EquationDependency:
    target: int
    sources: tuple[int, ...]
    expression: str
    lines: tuple[int, ...]


@dataclass(frozen=True)
class ODETermDependency:
    """One syntactic flux term, expanded through the same symbolic interpreter.

    Names/parameter keys are provenance anchors, not inferred biological labels.
    """
    target: int
    sources: tuple[int, ...]
    expression: str
    names: tuple[str, ...]
    parameters: tuple[str, ...]
    line: int


def _flux_terms(node):
    """Distribute source-level +/- and products, never a denominator or helper."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        left, right = _flux_terms(node.left), _flux_terms(node.right)
        if isinstance(node.op, ast.Sub):
            right = [ast.copy_location(ast.UnaryOp(op=ast.USub(), operand=t), node) for t in right]
        return left + right
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        return [ast.copy_location(ast.BinOp(left=a, op=ast.Mult(), right=b), node)
                for a in _flux_terms(node.left) for b in _flux_terms(node.right)]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return [ast.copy_location(ast.UnaryOp(op=node.op, operand=t), node) for t in _flux_terms(node.operand)]
    return [node]


@dataclass(frozen=True)
class EventDependency:
    event_id: int
    name: str
    guard: str
    guard_sources: tuple[int, ...]
    direction: int
    terminal: bool
    guard_lines: tuple[int, ...]
    resets: tuple[EquationDependency, ...]


@dataclass(frozen=True)
class YeastGraph:
    variable_names: tuple[str, ...]
    equations: tuple[EquationDependency, ...]
    events: tuple[EventDependency, ...]
    source_path: str
    source_sha256: str
    source_statements: tuple[tuple[int, str], ...]
    schema_version: int = SCHEMA_VERSION
    terms: tuple[ODETermDependency, ...] = ()

    def adjacency(self, view: str, *, self_loops: str = "physical") -> np.ndarray:
        """Return a fresh unsigned bool matrix, always A[target, source].

        physical: keep only self-dependencies found in source;
        none: drop the diagonal; all: explicitly add neural-network self-loops.
        event_trigger projects guards onto explicitly reset variables, NOT all
        effects of event timing. reset_value tracks reads by reset expressions.
        """
        if view not in VIEWS:
            raise ValueError(f"view must be one of {VIEWS}, got {view!r}")
        if self_loops not in ("physical", "none", "all"):
            raise ValueError("self_loops must be 'physical', 'none', or 'all'")
        matrix = np.zeros((len(self.variable_names),) * 2, dtype=bool)
        if view in ("continuous", "hybrid"):
            for equation in self.equations:
                matrix[equation.target, list(equation.sources)] = True
        if view != "continuous":
            for event in self.events:
                for reset in event.resets:
                    if view in ("event_trigger", "event", "hybrid"):
                        matrix[reset.target, list(event.guard_sources)] = True
                    if view in ("reset_value", "event", "hybrid"):
                        matrix[reset.target, list(reset.sources)] = True
        if self_loops != "physical":
            np.fill_diagonal(matrix, self_loops == "all")
        return matrix

    def summary(self) -> dict:
        continuous = self.adjacency("continuous", self_loops="none")
        hybrid = self.adjacency("hybrid", self_loops="none")
        return {
            "schema_version": self.schema_version,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "num_variables": len(self.variable_names),
            "orientation": "target_source",
            "sign_policy": "not_encoded; local signs may depend on state/parameters",
            "nonself_edges": {v: int(self.adjacency(v, self_loops="none").sum()) for v in VIEWS},
            "event_added_nonself_edges": int((hybrid & ~continuous).sum()),
            "events": [{"id": e.event_id, "name": e.name, "guard": e.guard,
                        "direction": e.direction, "terminal": e.terminal,
                        "guard_variables": [self.variable_names[i] for i in e.guard_sources],
                        "reset_variables": [self.variable_names[r.target] for r in e.resets]}
                       for e in self.events],
            "limits": "Potential structure only; not correlation, finite-horizon influence, or saltation support",
        }

    def to_dict(self) -> dict:
        """One optional JSON bundle of records/provenance, not redundant matrices."""
        return {**self.summary(), "variable_names": list(self.variable_names),
                "equations": [asdict(e) for e in self.equations],
                "event_records": [asdict(e) for e in self.events],
                "source_statements": dict(self.source_statements),
                "ode_terms": [asdict(t) for t in self.terms]}


@dataclass(frozen=True)
class _Value:
    expr: object
    lines: frozenset[int] = frozenset()


class _Extractor:
    def __init__(self, source: str, path: str):
        self.source, self.path = source, path
        self.terms = []
        try:
            self.tree = ast.parse(source, filename=path)
        except SyntaxError as exc:
            raise GraphExtractionError(f"{path}: invalid Python source: {exc}") from exc
        functions = [n for n in self.tree.body if isinstance(n, ast.FunctionDef)]
        self.functions = {n.name: n for n in functions}
        if len(self.functions) != len(functions):
            raise GraphExtractionError("Duplicate top-level function names")
        for name in ("eqns", "simulate_mutant"):
            if name not in self.functions:
                raise GraphExtractionError(f"Missing source function {name}")
        self.used_functions: set[str] = set()
        self.call_stack: list[str] = []
        eqns = self.functions["eqns"]
        tuples = [n for n in eqns.body if isinstance(n, ast.Assign)
                  and len(n.targets) == 1 and isinstance(n.targets[0], ast.Tuple)
                  and isinstance(n.value, ast.Name) and n.value.id == "y"]
        if len(tuples) != 1 or not all(isinstance(n, ast.Name) for n in tuples[0].targets[0].elts):
            self.fail(eqns, "Expected one named state tuple unpacked from y")
        self.unpack = tuples[0]
        self.names = tuple(n.id for n in self.unpack.targets[0].elts)
        if len(set(self.names)) != len(self.names):
            self.fail(self.unpack, "Duplicate state names")
        self.symbols = tuple(sp.Symbol("state__" + n, real=True) for n in self.names)
        self.state_indices = {s: i for i, s in enumerate(self.symbols)}
        bindings = [n for n in self.tree.body if isinstance(n, ast.Assign)
                    and len(n.targets) == 1 and isinstance(n.targets[0], ast.Tuple)
                    and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name)
                    and n.value.func.id == "range"]
        bindings = [n for n in bindings if [getattr(t, "id", None) for t in n.targets[0].elts]
                    == ["n" + name for name in self.names]]
        if len(bindings) != 1:
            self.fail(eqns, "State index tuple must agree with eqns state order")
        self.binding = bindings[0]
        if (len(self.binding.value.args) != 1 or self.binding.value.keywords
                or ast.literal_eval(self.binding.value.args[0]) != len(self.names)):
            self.fail(self.binding, "State index range must match the number of states")

    def fail(self, node, message):
        raise GraphExtractionError(f"{self.path}:{getattr(node, 'lineno', '?')}: {message}")

    def text(self, node):
        return ast.get_source_segment(self.source, node) or ast.unparse(node)

    def sources(self, value):
        # Expressions are normalized symbolically, never evaluated at WT/real parameters.
        expression = sp.factor_terms(sp.expand_mul(value.expr))
        return tuple(sorted(self.state_indices[s] for s in expression.free_symbols
                            if s in self.state_indices))

    @staticmethod
    def truth(expr):
        if isinstance(expr, Boolean) and not isinstance(expr, sp.Symbol):
            return expr
        return sp.Ne(expr, 0)

    def combine(self, expr, *values, line=None):
        lines = frozenset().union(*(v.lines for v in values))
        return _Value(expr, lines | ({line} if line is not None else set()))

    def merge(self, condition, yes, no):
        if yes == no:
            return yes
        return self.combine(sp.Piecewise((yes.expr, self.truth(condition.expr)),
                                        (no.expr, True)), condition, yes, no)

    def evaluate(self, node, env, states=None):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, bool)):
            expr = (sp.true if node.value else sp.false) if isinstance(node.value, bool) else sp.Rational(str(node.value))
            return _Value(expr)
        if isinstance(node, ast.Name):
            if node.id not in env:
                self.fail(node, f"Unresolved name {node.id!r}")
            return env[node.id]
        if isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Name) and node.value.id in ("p", "ovr"):
                if not isinstance(node.slice, ast.Constant) or not isinstance(node.slice.value, str):
                    self.fail(node, "Parameter keys must be string literals")
                return _Value(sp.Symbol(f"{node.value.id}__{node.slice.value}", real=True))
            if isinstance(node.value, ast.Name) and node.value.id in ("y", "y0") and states is not None:
                indices = self.indices(node.slice)
                if len(indices) != 1:
                    self.fail(node, "Scalar expression cannot read a state slice")
                return states[indices[0]]
            self.fail(node, "Unsupported subscript")
        if isinstance(node, ast.BinOp):
            a, b = self.evaluate(node.left, env, states), self.evaluate(node.right, env, states)
            operations = {ast.Add: lambda: a.expr + b.expr, ast.Sub: lambda: a.expr - b.expr,
                          ast.Mult: lambda: a.expr * b.expr, ast.Div: lambda: a.expr / b.expr,
                          ast.Pow: lambda: a.expr ** b.expr}
            if type(node.op) not in operations:
                self.fail(node, "Unsupported arithmetic operator")
            return self.combine(operations[type(node.op)](), a, b)
        if isinstance(node, ast.UnaryOp):
            a = self.evaluate(node.operand, env, states)
            if isinstance(node.op, ast.USub):
                return self.combine(-a.expr, a)
            if isinstance(node.op, ast.UAdd):
                return a
            if isinstance(node.op, ast.Not):
                return self.combine(sp.Not(self.truth(a.expr)), a)
            self.fail(node, "Unsupported unary operator")
        if isinstance(node, ast.Compare):
            if len(node.ops) != 1:
                self.fail(node, "Chained comparisons are not supported")
            a, b = self.evaluate(node.left, env, states), self.evaluate(node.comparators[0], env, states)
            operations = {ast.Eq: sp.Eq, ast.NotEq: sp.Ne, ast.Lt: sp.Lt,
                          ast.LtE: sp.Le, ast.Gt: sp.Gt, ast.GtE: sp.Ge}
            if type(node.ops[0]) not in operations:
                self.fail(node, "Unsupported comparison")
            return self.combine(operations[type(node.ops[0])](a.expr, b.expr), a, b)
        if isinstance(node, ast.BoolOp):
            values = [self.evaluate(n, env, states) for n in node.values]
            operation = sp.And if isinstance(node.op, ast.And) else sp.Or
            return self.combine(operation(*(self.truth(v.expr) for v in values)), *values)
        if isinstance(node, ast.IfExp):
            condition = self.evaluate(node.test, env, states)
            condition = self.combine(condition.expr, condition, line=node.lineno)
            self.check_condition(condition, node)
            return self.merge(condition, self.evaluate(node.body, env, states),
                              self.evaluate(node.orelse, env, states))
        if isinstance(node, ast.Call):
            if (isinstance(node.func, ast.Attribute) and node.func.attr == "get"
                    and isinstance(node.func.value, ast.Name) and node.func.value.id in ("p", "ovr")):
                if (node.keywords or len(node.args) not in (1, 2) or not isinstance(node.args[0], ast.Constant)
                        or not isinstance(node.args[0].value, str)):
                    self.fail(node, "Unsupported parameter get()")
                if len(node.args) == 2 and (not isinstance(node.args[1], ast.Constant)
                        or node.args[1].value is not None
                        and not isinstance(node.args[1].value, (int, float, bool))):
                    self.fail(node, "Parameter get() defaults must be literal scalars")
                # Unknown presence/default does not remove potential parameter-dependent branches.
                return _Value(sp.Symbol(f"{node.func.value.id}__{node.args[0].value}", real=True))
            if node.keywords:
                self.fail(node, "Keyword calls are not supported in scalar expressions")
            values = [self.evaluate(n, env, states) for n in node.args]
            name = ast.unparse(node.func)
            operations = {"np.log": sp.log, "np.exp": sp.exp, "np.sqrt": sp.sqrt,
                          "np.maximum": sp.Max, "np.minimum": sp.Min, "abs": sp.Abs}
            if name in operations:
                if len(values) != (2 if name in ("np.maximum", "np.minimum") else 1):
                    self.fail(node, f"Unsupported argument count for {name}")
                return self.combine(operations[name](*(v.expr for v in values)), *values)
            if isinstance(node.func, ast.Name) and name in self.functions:
                return self.scalar_function(name, values)
            self.fail(node, f"Unsupported scalar call {name!r}")
        self.fail(node, f"Unsupported expression {type(node).__name__}")

    def check_condition(self, condition, node):
        if set(getattr(condition.expr, "free_symbols", ())) & set(self.symbols):
            self.fail(node, "State-dependent Python branches require an explicit mode/event contract")

    def scalar_function(self, name, values):
        function = self.functions[name]
        args = function.args
        if (function.decorator_list or args.vararg or args.kwarg or args.defaults or args.kwonlyargs or args.posonlyargs
                or len(args.args) != len(values) or name in self.call_stack):
            self.fail(function, f"Unsupported signature or recursive scalar helper {name}")
        self.used_functions.add(name)
        self.call_stack.append(name)
        env = dict(zip((a.arg for a in args.args), values))
        result = None
        for statement in function.body:
            if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant) and isinstance(statement.value.value, str):
                continue
            if result is not None:
                self.fail(statement, "Statements after helper return are not supported")
            if isinstance(statement, ast.Assign) and all(isinstance(t, ast.Name) for t in statement.targets):
                value = self.evaluate(statement.value, env)
                for target in statement.targets:
                    env[target.id] = self.combine(value.expr, value, line=statement.lineno)
            elif isinstance(statement, ast.Return):
                value = self.evaluate(statement.value, env)
                result = self.combine(value.expr, value, line=statement.lineno)
            else:
                self.fail(statement, f"Unsupported helper statement {type(statement).__name__}")
        self.call_stack.pop()
        if result is None:
            self.fail(function, "Scalar helper must return an expression")
        return result

    def indices(self, node):
        if isinstance(node, ast.Name) and node.id in ("n" + n for n in self.names):
            return [self.names.index(node.id[1:])]
        if isinstance(node, ast.Constant) and type(node.value) is int:
            indices = [node.value]
        elif isinstance(node, ast.Slice) and node.step is None:
            try:
                start = 0 if node.lower is None else ast.literal_eval(node.lower)
                stop = len(self.names) if node.upper is None else ast.literal_eval(node.upper)
            except (ValueError, TypeError):
                self.fail(node, "State slice bounds must be literal integers")
            if type(start) is not int or type(stop) is not int or not 0 <= start < stop <= len(self.names):
                self.fail(node, "Invalid state slice")
            indices = list(range(start, stop))
        else:
            self.fail(node, "Unsupported state index")
        if any(i < 0 or i >= len(self.names) for i in indices):
            self.fail(node, "State index outside state vector")
        return indices

    def assignments(self, statements, env, states, outputs, *, reset=False):
        """Track scalar SSA-like values and explicitly written output channels."""
        for statement in statements:
            if statement is self.unpack:
                continue
            if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant) and isinstance(statement.value.value, str):
                continue
            if isinstance(statement, ast.Return) and not reset:
                if not isinstance(statement.value, ast.Name) or statement.value.id != "dydt":
                    self.fail(statement, "ODE must return dydt")
                continue
            if isinstance(statement, ast.Expr) and reset:
                call = statement.value
                if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == "append"
                        and isinstance(call.func.value, ast.Name) and call.func.value.id in ("bd", "md")):
                    if len(call.args) != 1 or call.keywords:
                        self.fail(statement, "Unsupported observer append signature")
                    self.evaluate(call.args[0], env, states)  # Also reject nested mutations/calls.
                    continue  # Observer logs do not mutate state or control the resets.
                self.fail(statement, "Unsupported reset side effect")
            if isinstance(statement, ast.If):
                condition = self.evaluate(statement.test, env, states)
                condition = self.combine(condition.expr, condition, line=statement.lineno)
                self.check_condition(condition, statement)
                yes_env, no_env = dict(env), dict(env)
                yes_states, no_states = list(states), list(states)
                yes_out, no_out = dict(outputs), dict(outputs)
                self.assignments(statement.body, yes_env, yes_states, yes_out, reset=reset)
                self.assignments(statement.orelse, no_env, no_states, no_out, reset=reset)
                for key in yes_env.keys() | no_env.keys():
                    if key not in yes_env or key not in no_env:
                        self.fail(statement, f"Local {key!r} not defined on both branches")
                    env[key] = self.merge(condition, yes_env[key], no_env[key])
                for i in set(yes_out) | set(no_out):
                    if i not in yes_out or i not in no_out:
                        self.fail(statement, "Output not defined on both branches")
                    outputs[i] = self.merge(condition, yes_out[i], no_out[i])
                states[:] = [self.merge(condition, a, b) for a, b in zip(yes_states, no_states)]
                continue
            if isinstance(statement, ast.Assign):
                # Only this exact initialization is ignored, not arbitrary numpy calls.
                if (len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name)
                        and statement.targets[0].id == "dydt"):
                    call = statement.value
                    if (reset or not isinstance(call, ast.Call) or ast.unparse(call.func) != "np.zeros"
                            or len(call.args) != 1 or call.keywords
                            or ast.literal_eval(call.args[0]) != len(self.names)):
                        self.fail(statement, "Unsupported dydt initialization")
                    continue
                value = self.evaluate(statement.value, env, states)
                value = self.combine(value.expr, value, line=statement.lineno)
                targets = statement.targets
            elif isinstance(statement, ast.AugAssign) and reset:
                previous = self.evaluate(statement.target, env, states)
                increment = self.evaluate(statement.value, env, states)
                if isinstance(statement.op, ast.Add):
                    expr = previous.expr + increment.expr
                elif isinstance(statement.op, ast.Sub):
                    expr = previous.expr - increment.expr
                else:
                    self.fail(statement, "Unsupported augmented reset operator")
                value = self.combine(expr, previous, increment, line=statement.lineno)
                targets = [statement.target]
            else:
                self.fail(statement, f"Unsupported statement {type(statement).__name__}")
            for target in targets:
                if isinstance(target, ast.Name):
                    if target.id in self.names or target.id in ("n" + n for n in self.names) or target.id in ("y", "y0", "p", "ovr"):
                        self.fail(target, "Rebinding a state/parameter input is not supported")
                    env[target.id] = value
                elif (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)
                      and target.value.id == ("y0" if reset else "dydt")):
                    for i in self.indices(target.slice):
                        outputs[i] = value
                        if reset:
                            states[i] = value
                        elif isinstance(statement, ast.Assign):
                            for term in _flux_terms(statement.value):
                                term_value = self.evaluate(term, env, states)
                                names = tuple(sorted({n.id for n in ast.walk(term) if isinstance(n, ast.Name)}))
                                parameters = tuple(sorted({n.slice.value for n in ast.walk(term)
                                    if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)
                                    and n.value.id == 'p' and isinstance(n.slice, ast.Constant)
                                    and isinstance(n.slice.value, str)}))
                                self.terms.append(ODETermDependency(
                                    i, self.sources(term_value), ast.unparse(term), names,
                                    parameters, statement.lineno))
                else:
                    self.fail(target, "Unsupported assignment target")

    def extract(self):
        self.used_functions.add("eqns")
        function = self.functions["eqns"]
        if (function.decorator_list or [a.arg for a in function.args.args] != ["t", "y", "p"]
                or function.args.vararg or function.args.kwarg or function.args.defaults
                or function.args.kwonlyargs or function.args.posonlyargs):
            self.fail(function, "ODE signature must be eqns(t, y, p)")
        returns = [n for n in ast.walk(function) if isinstance(n, ast.Return)]
        initializers = [n for n in ast.walk(function) if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == "dydt" for t in n.targets)]
        if (len(returns) != 1 or function.body[-1] is not returns[0]
                or len(initializers) != 1 or initializers[0] not in function.body):
            self.fail(function, "Expected one top-level dydt initialization and one final return")
        env = {name: _Value(symbol) for name, symbol in zip(self.names, self.symbols)}
        env["t"] = _Value(sp.Symbol("time", real=True))
        states = [_Value(s) for s in self.symbols]
        outputs = {}
        self.assignments(self.functions["eqns"].body, env, states, outputs)
        if set(outputs) != set(range(len(self.names))):
            self.fail(self.functions["eqns"], "Every ODE channel must be assigned explicitly")
        equations = tuple(EquationDependency(i, self.sources(outputs[i]),
                          self.source.splitlines()[max(outputs[i].lines) - 1].strip(),
                          tuple(sorted(outputs[i].lines))) for i in range(len(self.names)))

        simulation = self.functions["simulate_mutant"]
        calls = [n for n in ast.walk(simulation) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == "solve_ivp"]
        if len(calls) != 1:
            self.fail(simulation, "Expected one solve_ivp call defining event order")
        if not calls[0].args or not isinstance(calls[0].args[0], ast.Name) or calls[0].args[0].id != "eqns":
            self.fail(calls[0], "Registered ODE must be eqns")
        event_lists = [k.value for k in calls[0].keywords if k.arg == "events"]
        if (len(event_lists) != 1 or not isinstance(event_lists[0], (ast.List, ast.Tuple))
                or not all(isinstance(n, ast.Name) for n in event_lists[0].elts)):
            self.fail(calls[0], "Expected an explicit named event list")
        event_names = [n.id for n in event_lists[0].elts]
        if len(set(event_names)) != len(event_names) or not event_names:
            self.fail(calls[0], "Event names must be nonempty and unique")

        # The reset branch ids must really mean the 1-based solve_ivp event order.
        selectors = [n for n in ast.walk(simulation) if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == "ie" for t in n.targets)]
        if len(selectors) != 1:
            self.fail(simulation, "Expected one event-id selector")
        selector = selectors[0]
        call = selector.value
        if (not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name)
                or call.func.id != "next" or not call.args
                or not isinstance(call.args[0], ast.GeneratorExp) or call.keywords):
            self.fail(selector, "Event ids must be selected from enumerate(sol.t_events)")
        generator = call.args[0]
        if len(generator.generators) != 1:
            self.fail(selector, "Unsupported event-id comprehension")
        comprehension = generator.generators[0]
        if (not isinstance(comprehension.target, ast.Tuple) or len(comprehension.target.elts) != 2
                or not all(isinstance(t, ast.Name) for t in comprehension.target.elts)
                or ast.unparse(comprehension.iter) != "enumerate(sol.t_events)"
                or comprehension.is_async):
            self.fail(selector, "Event ids must enumerate sol.t_events in registration order")
        index_name = comprehension.target.elts[0].id
        expected_id = ast.parse(f"{index_name} + 1", mode="eval").body
        if ast.dump(generator.elt) != ast.dump(expected_id):
            self.fail(selector, "Reset dispatch requires 1-based event ids")
        time_name = comprehension.target.elts[1].id
        allowed_filters = [f"len({time_name}) > 0",
                           f"len({time_name}) > 0 and np.isclose({time_name}[-1], t0)"]
        if (len(comprehension.ifs) != 1 or ast.dump(comprehension.ifs[0]) not in
                [ast.dump(ast.parse(text, mode="eval").body) for text in allowed_filters]):
            self.fail(selector, "Unsupported event filtering/priority logic")

        def dispatch_id(node):
            test = node.test if isinstance(node, ast.If) else None
            if (isinstance(test, ast.Compare) and isinstance(test.left, ast.Name) and test.left.id == "ie"
                    and len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq)
                    and isinstance(test.comparators[0], ast.Constant)):
                return test.comparators[0].value
            return None

        roots = [n for n in ast.walk(simulation) if dispatch_id(n) == 1]
        if len(roots) != 1:
            self.fail(simulation, "Expected one ie==1 reset dispatch")
        branches = {}
        node = roots[0]
        while node is not None:
            event_id = dispatch_id(node)
            if type(event_id) is not int or event_id in branches:
                self.fail(node, "Invalid reset dispatch")
            branches[event_id] = node.body
            if not node.orelse:
                break
            if len(node.orelse) != 1 or not isinstance(node.orelse[0], ast.If):
                self.fail(node, "Unsupported default reset branch")
            node = node.orelse[0]
        if set(branches) != set(range(1, len(event_names) + 1)):
            self.fail(roots[0], "Reset dispatch and event registration disagree")

        event_records, event_attributes = [], []
        for event_id, name in enumerate(event_names, 1):
            if name not in self.functions:
                self.fail(calls[0], f"Missing event function {name}")
            function = self.functions[name]
            if (function.decorator_list or [a.arg for a in function.args.args] != ["t", "y", "p"]
                    or function.args.vararg or function.args.kwarg or function.args.defaults
                    or function.args.kwonlyargs or function.args.posonlyargs):
                self.fail(function, "Unsupported event signature")
            self.used_functions.add(name)
            returns = [n for n in function.body if isinstance(n, ast.Return)]
            if len(returns) != 1 or len(function.body) != 1:
                self.fail(function, "Event guard must be one return expression")
            guard = self.evaluate(returns[0].value, {"t": env["t"]}, states)
            attrs = {}
            for st in self.tree.body:
                if isinstance(st, ast.Assign):
                    for target in st.targets:
                        if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == name:
                            if target.attr not in ("direction", "terminal") or target.attr in attrs:
                                self.fail(st, "Unsupported or duplicate event attribute")
                            attrs[target.attr] = ast.literal_eval(st.value)
                            event_attributes.append(st)
            if (set(attrs) != {"direction", "terminal"} or attrs["terminal"] is not True
                    or type(attrs["direction"]) is not int or attrs["direction"] not in (-1, 0, 1)):
                self.fail(function, "Expected explicit event direction and terminal=True for reset dispatch")
            reset_env = {flag: _Value(sp.Symbol("control__" + flag, real=True))
                         for flag in ("is_mad2_ko", "is_bub2_ko")}
            reset_states, reset_outputs = [_Value(s) for s in self.symbols], {}
            self.assignments(branches[event_id], reset_env, reset_states, reset_outputs, reset=True)
            resets = tuple(EquationDependency(i, self.sources(value), str(value.expr),
                           tuple(sorted(value.lines))) for i, value in sorted(reset_outputs.items()))
            event_records.append(EventDependency(event_id, name, self.text(returns[0].value),
                                 self.sources(guard), attrs["direction"], attrs["terminal"],
                                 tuple(sorted(guard.lines | {returns[0].lineno})), resets))

        protected = self.used_functions | {"n" + n for n in self.names}
        for statement in self.tree.body:
            if statement is self.binding:
                continue
            targets = statement.targets if isinstance(statement, ast.Assign) else (
                [statement.target] if isinstance(statement, (ast.AnnAssign, ast.AugAssign)) else [])
            for target in targets:
                if any(isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
                       and n.id in protected for n in ast.walk(target)):
                    self.fail(statement, "Rebinding a relevant function/state index is unsupported")

        kernel = [self.binding, *(self.functions[name] for name in sorted(self.used_functions)),
                  event_lists[0], selector, roots[0], *event_attributes]
        fingerprint = hashlib.sha256(json.dumps([ast.dump(n, include_attributes=False) for n in kernel],
                                                separators=(",", ":")).encode()).hexdigest()
        used_lines = set()
        for equation in equations:
            used_lines.update(equation.lines)
        for event in event_records:
            used_lines.update(event.guard_lines)
            for reset in event.resets:
                used_lines.update(reset.lines)
        source_lines = self.source.splitlines()
        statements = tuple((i, source_lines[i - 1].strip()) for i in sorted(used_lines))
        return YeastGraph(self.names, equations, tuple(event_records), self.path, fingerprint, statements,
                          terms=tuple(self.terms))


@lru_cache(maxsize=4)
def _build_cached(source: str, path: str) -> YeastGraph:
    try:
        return _Extractor(source, path).extract()
    except GraphExtractionError:
        raise
    except (ValueError, TypeError, KeyError, NotImplementedError) as exc:
        raise GraphExtractionError(f"{path}: unsupported or inconsistent symbolic source: {exc}") from exc


def build_yeast_graph(source_path: str | Path | None = None) -> YeastGraph:
    """Build/read an immutable graph; source edits invalidate the in-memory cache.

    Paths are relative to this module by default, not the process working dir.
    No on-disk artifact is read or written, and no source Python is executed.
    """
    path = (DEFAULT_SOURCE if source_path is None else Path(source_path)).resolve()
    return _build_cached(path.read_text(encoding="utf-8"), str(path))
