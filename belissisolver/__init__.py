from __future__ import annotations

import abc
import functools
import itertools
import math
import random
import string
import typing
from collections import defaultdict


def parens_aware_split(txt: str, split_char: str) -> list[str]:
    groups = [""]
    parens_level = 0
    for char in txt:
        if parens_level == 0 and char == split_char:
            groups.append("")
            continue

        groups[-1] += char

        if char == "(":
            parens_level += 1
        elif char == ")":
            parens_level -= 1

    return groups


def has_top_level_parens(txt: str) -> bool:
    if not txt.startswith("(") or not txt.endswith(")"):
        return False

    parens_level = 0
    for char in txt[:-1]:  # excluding the last char, that one should always have a parens_level=0
        if char == "(":
            parens_level += 1
        elif char == ")":
            parens_level -= 1

        if parens_level == 0:
            # if it drops to zero, there are no top level parens
            return False

    return True


def remove_top_level_parens(txt: str) -> str:
    txt = txt.strip()

    while has_top_level_parens(txt):
        txt = txt[1:-1]  # remove outermost chars since they are the (parens)
        txt = txt.strip()

    return txt


class ParsingException(Exception): ...


class DirectParsingException(ParsingException): ...


class ExpressionEvaluationException(Exception): ...


class Expression(abc.ABC):
    class _FromArgsDesc:
        def __get__(self, instance, owner) -> typing.Callable[..., Expression]:
            if instance is None:
                return owner._from_args_class
            else:
                return functools.partial(owner._from_args_instance, instance)

    min_args: int | None = 0
    max_args: int | None = None
    args: tuple[Expression]

    is_commutative: bool = False
    is_associative: bool = False
    is_idempotent: bool = False
    unit_element: Expression

    def __init__(self, *args: Expression):
        self.args = args

    from_args = _FromArgsDesc()

    @classmethod
    def _from_args(cls, *args: Expression) -> Expression:
        return cls(*args)

    @classmethod
    def _from_args_class(cls, *args: Expression) -> Expression:
        assert cls.max_args != 0, "Use instance.from_args() instead."

        if cls.is_commutative and len(args) == 1:  # TODO: Is this condition correct?
            return args[0]
        else:
            return cls._from_args(*args)

    def _from_args_instance(self, *args: Expression) -> Expression:
        if self.max_args == 0:
            return self
        else:
            return self._from_args_class(*args)

    @abc.abstractmethod
    def to_str_human(self, parent: Expression | None = None):
        ...

    @abc.abstractmethod
    def to_str_formal(self) -> str:
        ...

    def to_str_full(self, prune_no_args: bool = True) -> str:
        if prune_no_args and not self.args:
            return self.to_str_formal()
        else:
            return f"{self.__class__.__name__}({', '.join(arg.to_str_full() for arg in self.args)})"

    def eval(self) -> float:
        raise ExpressionEvaluationException(f"Can't evaluate {self}.")

    @classmethod
    def parse(cls, expr: str) -> "Expression":
        if expr.startswith(cls.__name__):
            expr = expr[len(cls.__name__):]
        else:
            raise DirectParsingException(f"Can't parse {expr!r} as {cls.__name__}.")

        if not expr.startswith("(") or not expr.endswith(")"):
            raise DirectParsingException(f"Can't parse {expr!r} as {cls.__name__}.")

        expr = expr[1:-1]

        return cls.from_args(*(Expression.from_str(arg) for arg in parens_aware_split(expr, ",")))

    @staticmethod
    def from_str(expr: str) -> "Expression":
        expr = remove_top_level_parens(expr)

        types = (
            _Out, _NthArg,
            Logarithm, Piecewise,
            _Differentiate, _Reduce, _Substitute,
            LogicalOr, LogicalAnd, LogicalNot, LessThan, Equality,
            Sum, Difference,
            Product, Quotient,
            Power,
            Value, Boolean, Variable
        )

        errors = []
        for expr_type in types:
            try:
                return expr_type.parse(expr)
            except (DirectParsingException, ParsingException) as e:
                errors.append(e)

        errors.sort(key=lambda error: isinstance(error, DirectParsingException))

        raise ParsingException(f"Could not parse expression {expr!r}.\n" +
                               "".join(f" - {error_msg}\n" for error_msg in errors))

    @property
    def is_constant_value(self):
        if not self.args:
            return True
        else:
            return all(arg.is_constant_value for arg in self.args)

    @property
    def is_indeterminate(self) -> BooleanExpression:
        return Boolean(False)

    def par_diff(self, i: int, var: Expression) -> Expression:
        raise NotImplementedError(f"Unknown partial derivative with respect to {i + 1}. argument of {self}.")

    def inverted(self, i: int, result: Expression) -> typing.Collection[Expression]:
        raise NotImplementedError(f"Can't invert this expression using {i + 1}. argument of {self}.")

    def doit(self) -> Expression:
        return self.from_args(*(arg.doit() for arg in self.args))

    def walk(self) -> typing.Iterable[Expression]:
        yield self

        for arg in self.args:
            yield from arg.walk()

    def __contains__(self, item):
        return item == self or any(item in arg for arg in self.args)

    def __eq__(self, other):
        if self.__class__ != other.__class__:
            return False

        if len(self.args) != len(other.args):
            return False

        if self.is_commutative:
            if set(self.args) == set(other.args):
                return True
        else:
            return all((arg1 == arg2) for arg1, arg2 in zip(self.args, other.args))

    def __str__(self):
        return self.to_str_formal()

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash((self.__class__.__name__, tuple(self.args)))


class BooleanExpression(Expression): pass


class Variable(Expression):
    max_args = 0
    forbidden_chars = "^*+-/(). "

    def __init__(self, symbol: str):
        super().__init__()
        self.symbol = symbol

    @classmethod
    def parse(cls, expr: str) -> "Expression":
        if any([char in cls.forbidden_chars for char in expr]):
            raise ParsingException(f"Symbol contains invalid chars: {expr!r}.")

        if not expr:
            raise ParsingException("Can't create an empty Symbol.")

        return Variable(expr)

    def eval(self) -> float:
        raise ExpressionEvaluationException(f"Can't evaluate unresolved variable {self.symbol!r}.")

    def to_str_formal(self) -> str:
        return self.symbol

    def to_str_human(self, parent: Expression | None = None):
        return self.symbol

    @property
    def is_constant_value(self):
        return False

    def __contains__(self, item):
        return self == item

    def __eq__(self, other):
        if not isinstance(other, Variable):
            return False

        return self.symbol == other.symbol

    def __hash__(self):
        return hash(self.symbol)


class UniqueVariable(Variable):
    def __init__(self):
        super().__init__(f"<U{id(self)}>")

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return hash(id(self))


class Value(Expression):
    max_args = 0

    def __init__(self, value: float):
        super().__init__()
        self.value = value

    @classmethod
    def parse(cls, expr: str) -> "Expression":
        try:
            return Value(int(expr))
        except ValueError:
            try:
                return Value((float(expr)))
            except ValueError as e:
                raise DirectParsingException(f"Can't parse {expr!r} as a number.") from e

    def eval(self) -> float:
        return self.value

    def to_str_formal(self) -> str:
        return f"{self.value}"

    def to_str_human(self, parent: Expression | None = None):
        out = self.to_str_formal()

        if out.startswith("-") and parent is not None:
            return f"({out})"
        else:
            return out

    def __contains__(self, item):
        return self == item

    def __eq__(self, other):
        if not isinstance(other, Value):
            return False

        return self.value == other.value

    def __hash__(self):
        return hash(self.value)


class Constant(Value):
    max_args = 0

    def __init__(self, symbol: str, value: float):
        super().__init__(value)
        self.symbol = symbol

    def to_str_formal(self) -> str:
        return self.symbol


Constant.pi = Constant("π", 3.141592653589793)
Constant.e = Constant("𝑒", 2.718281828459045)


class Boolean(Value, BooleanExpression):
    @classmethod
    def parse(cls, expr: str) -> "Expression":
        if expr == "True":
            return Boolean(True)
        elif expr == "False":
            return Boolean(False)

        raise DirectParsingException(f"Can't parse {expr!r} as a boolean.")


class _OperatorExpression(Expression, abc.ABC):
    operator: str

    @classmethod
    def parse(cls, expr: str) -> "Expression":
        str_args = parens_aware_split(expr, cls.operator)

        if cls.max_args and len(str_args) > cls.max_args:
            raise DirectParsingException(f"Could not parse {expr!r} as {cls.__name__}.")

        if len(str_args) > 1:
            return cls.from_args(*(Expression.from_str(arg) for arg in str_args))
        else:
            raise DirectParsingException(f"Can't parse {expr!r} as {cls.__name__}.")

    def to_str_formal(self) -> str:
        return "(" + f" {self.operator} ".join([expr.to_str_formal() for expr in self.args]) + ")"

    def _to_str_human_no_parens(self, parent: Expression):
        return f" {self.operator} ".join([expr.to_str_human(self) for expr in self.args])

    def _to_str_human_parens(self, parent: Expression):
        return "(" + self._to_str_human_no_parens(self) + ")"

    def to_str_human(self, parent: Expression | None = None):
        return "(" + f" {self.operator} ".join([expr.to_str_human(self) for expr in self.args]) + ")"


class Sum(_OperatorExpression):
    operator = "+"
    unit_element = Value(0)
    is_commutative = True
    is_associative = True

    def factor_out(self, expr: Expression):
        # 2 a + 3 a + a + b
        # a * (2 a / a + 3 a / a + a / a + b / a)
        # a (2 + 3 + 1 + b / a)

        sum_args = []

        for arg in self.args:
            sum_args.append(Quotient.from_args(arg, expr))

        return Product(expr, Sum.from_args(*sum_args))

    def to_str_human(self, parent: Expression | None = None):
        if parent is None or isinstance(parent, Sum):
            return self._to_str_human_no_parens(parent)
        else:
            return self._to_str_human_parens(parent)

    def eval(self) -> float:
        return sum([arg.eval() for arg in self.args])

    def par_diff(self, i: int, var: Expression) -> Expression:
        return Value(1)

    def inverted(self, i: int, result: Expression) -> tuple[Expression]:
        return Difference.from_args(result, *(arg for j, arg in enumerate(self.args) if j != i)),


class Product(_OperatorExpression):
    operator = "*"
    unit_element = Value(1)
    is_commutative = True
    is_associative = True

    def eval(self) -> float:
        product = 1
        for arg in self.args:
            product *= arg.eval()
        return product

    def to_str_human(self, parent: Expression | None = None):
        if parent is None or isinstance(parent, (Product, Sum)):
            return self._to_str_human_no_parens(parent)
        else:
            return self._to_str_human_parens(parent)

    def par_diff(self, i: int, var: Variable) -> Expression:
        return Product.from_args(*(arg for j, arg in enumerate(self.args) if j != i))

    def inverted(self, i: int, result: Expression) -> tuple[Expression]:
        return Quotient.from_args(result, *(arg for j, arg in enumerate(self.args) if j != i)),


class CantInit:
    def __init__(self, *args: Expression):
        raise ParsingException(
            f"{self.__class__.__name__}s are not meant to be instantiated. Use {self.__class__.__name__}.from_args() "
            f"to yield an equivalent Expression."
        )


class Quotient(_OperatorExpression, CantInit):
    operator = "/"
    min_args = 2

    @classmethod
    def _from_args(cls, *args: Expression) -> Expression:
        return Product(args[0], *[Power(arg, Value(-1)) for arg in args[1:]])

    def eval(self) -> float:
        ...


class Difference(_OperatorExpression, CantInit):
    operator = "-"
    min_args = 2

    @classmethod
    def _from_args(cls, *args: Expression) -> Sum:
        return Sum(args[0], *[Product(arg, Value(-1)) for arg in args[1:]])

    def eval(self) -> float:
        ...


class FunctionSyntaxExpression(Expression, abc.ABC):
    def to_str_human(self, parent: Expression | None = None):
        return f"{self.__class__.__name__}({', '.join(arg.to_str_human(self) for arg in self.args)})"

    def to_str_formal(self) -> str:
        return f"{self.__class__.__name__}({', '.join(arg.to_str_formal() for arg in self.args)})"


class IsEven(FunctionSyntaxExpression):
    # don't have resolve yet, so can't prove this :(

    max_args = 1

    def eval(self) -> float:
        return self.args[0].eval() % 2 == 0


class Piecewise(FunctionSyntaxExpression):
    is_commutative = False
    is_associative = False

    def eval(self) -> float:
        for arg, cond in self.iter_args():
            if cond.eval():
                return arg.eval()

        raise ExpressionEvaluationException

    def _from_args_instance(self, *args: Expression) -> Expression:
        assert len(args) % 2 == 0, "Piecewise expressions must have an even number of arguments."

        return super()._from_args_instance(*args)

    def iter_args(self) -> typing.Iterable[tuple[Expression, Expression]]:
        for i in range(0, len(self.args), 2):
            yield self.args[i], self.args[i + 1]

    def par_diff(self, i: int, var: Expression) -> Expression:
        if i % 2 != 0:
            raise NotImplementedError("Can't differentiate a piecewise function with respect to its condition.")

        return Piecewise.from_args(*zip(
            # TODO somehow inform par_diff of condition
            (arg.par_diff(i, var) for arg in self.args[::2]),
            self.args[1::2]
        ))

    def inverted(self, i: int, result: Expression) -> tuple[Expression]:
        raise NotImplementedError("Can't yet invert a piecewise function, sorry. :(")

    @property
    def is_indeterminate(self):
        # TODO: Maybe more strict?
        out = LogicalOr.from_args(
            LogicalAnd.from_args(*(arg.is_indeterminate for arg in self.args[::2])),
            LogicalAnd.from_args(*(arg.is_indeterminate for arg in self.args[1::2]))
        )
        return out


class Power(_OperatorExpression):
    operator = "^"
    max_args = 2

    @classmethod
    def parse(cls, expr: str) -> "Expression":
        return super().parse(expr)

    def eval(self) -> float:
        try:
            return self.args[0].eval() ** self.args[1].eval()
        except (OverflowError, ZeroDivisionError) as e:
            raise ExpressionEvaluationException(
                f"A mathematical error occurred during the evaluation of the expression {self}."
            ) from e

    @property
    def base(self):
        return self.args[0]

    @property
    def exponent(self):
        return self.args[1]

    @property
    def is_indeterminate(self) -> BooleanExpression:
        # noinspection PyTypeChecker
        return LogicalOr.from_args(
            LogicalAnd.from_args(
                Equality(self.exponent, Value(0)),
                Equality(self.base, Value(0))
            ),
            LogicalAnd.from_args(
                LessThan(self.exponent, Value(0)),
                Equality(self.base, Value(0))
            ),
            # TODO: More cases
        )

    def par_diff(self, i: int, var: Expression) -> Expression:
        if i == 0:
            return Product.from_args(
                self.exponent,
                Power.from_args(var, Difference.from_args(self.exponent, Value(1)))
            )
        else:
            return Product.from_args(
                self,
                Logarithm.from_args(Constant.e, self.base)
            )

    def inverted(self, i: int, result: Expression) -> tuple[Expression, ...]:
        if i == 0:
            # two solutions iff exponent is even
            sol = Power.from_args(result, Quotient.from_args(Value(1), self.exponent))
            if_even = Product.from_args(sol, Value(-1))

            return (
                sol,
                Piecewise.from_args(
                    if_even,
                    IsEven.from_args(self.exponent)
                )
            )
        else:
            return Logarithm.from_args(self.base, result),


class Logarithm(FunctionSyntaxExpression):
    is_commutative = False
    is_associative = False

    def eval(self) -> float:
        try:
            return math.log(self.argument.eval(), self.base.eval())
        except (ArithmeticError, ValueError) as e:
            raise ExpressionEvaluationException(
                f"A mathematical error occurred during the evaluation of the expression {self}."
            ) from e

    @property
    def base(self):
        return self.args[0]

    @property
    def argument(self):
        return self.args[1]

    @property
    def is_indeterminate(self):
        return LogicalOr.from_args(
            Equality(self.argument, Value(0)),
            LessThan(self.argument, Value(0))
            # TODO: Check
        )

    def par_diff(self, i: int, var: Variable) -> Expression:
        if i == 0:
            return Variable("__close_your_eyes_🙈")
        else:
            return Quotient.from_args(
                Value(1),
                Product.from_args(
                    self.argument,
                    Logarithm.from_args(Constant.e, self.base)
                )
            )

    def inverted(self, i: int, result: Expression) -> tuple[Expression]:
        if i == 0:
            raise NotImplementedError("Can't invert logarithm with respect to base.")
        else:
            return Power.from_args(self.base, result),


class Equality(_OperatorExpression, BooleanExpression):
    operator = "="
    min_args = 2
    max_args = 2
    is_commutative = False  # this should be True, but Eq is special
    is_associative = False
    unit_element = Boolean(True)

    def eval(self) -> float:
        return self.args[0].eval() == self.args[1].eval()

    @property
    def lhs(self):
        return self.args[0]

    @property
    def rhs(self):
        return self.args[1]


class LessThan(_OperatorExpression, BooleanExpression):
    operator = "<"
    min_args = 2
    max_args = 2
    is_commutative = False
    is_associative = False

    def eval(self) -> float:
        return self.args[0].eval() < self.args[1].eval()

    @property
    def lhs(self):
        return self.args[0]

    @property
    def rhs(self):
        return self.args[1]


class LogicalOr(_OperatorExpression, BooleanExpression):
    operator = "∨"
    is_commutative = True
    is_associative = True
    is_idempotent = True
    unit_element = Boolean(False)

    @classmethod
    def _from_args(cls, *args: Expression) -> Expression:
        if not args:
            return Boolean(False)
        else:
            return super()._from_args(*args)

    def eval(self) -> float:
        return any(arg.eval() for arg in self.args)


class LogicalAnd(_OperatorExpression, BooleanExpression, CantInit):
    # TODO: Maybe implement just as combination of OR and NOT

    operator = "∧"
    is_commutative = True
    is_associative = True
    is_idempotent = True
    unit_element = Boolean(True)

    def eval(self) -> float:
        return all(arg.eval() for arg in self.args)

    @classmethod
    def _from_args(cls, *args: Expression) -> Expression:
        return LogicalNot.from_args(LogicalOr.from_args(*(
            LogicalNot.from_args(arg) for arg in args
        )))


class LogicalNot(FunctionSyntaxExpression, BooleanExpression):
    min_args = 1
    max_args = 1

    def eval(self) -> float:
        return not self.args[0].eval()

    def to_str_formal(self) -> str:
        return f"¬({self.args[0].to_str_formal()})"

    def to_str_human(self, parent: Expression | None = None) -> str:
        return f"¬({self.args[0].to_str_human(self)})"

    def inverted(self, i: int, result: Expression) -> typing.Collection[Expression]:
        assert i == 0

        return LogicalNot.from_args(result),


def simplify_idempotent(expr: Expression) -> Expression:
    if not expr.is_idempotent:
        return expr

    return expr.from_args(*set(expr.args))


def simplify_associative(expr: Expression) -> Expression:
    if not expr.is_associative:
        return expr

    args = []

    for arg in expr.args:
        if isinstance(arg, expr.__class__):
            args += simplify_associative(arg).args
        else:
            args.append(arg)

    return expr.from_args(*args)


def simplify_commutative(expr: Expression) -> Expression:
    if not expr.is_commutative:
        return expr

    args: list[Expression] = []
    constants = []
    numbers = []

    assert expr.unit_element is not None

    for arg in expr.args:
        if arg != expr.unit_element:
            if isinstance(arg, Value):
                numbers.append(arg)
            elif arg.is_constant_value:
                constants.append(arg)
            else:
                args.append(arg)

    args += [simplify(expr.from_args(*(num for num in numbers)))] if numbers else []
    args += constants

    if len(args) == 0:
        expr = expr.unit_element
    else:
        expr = expr.from_args(*args)

    return expr


def simplify_abstract_arg_expr(expr: Expression) -> Expression:
    expr = expr.from_args(*[simplify(arg) for arg in expr.args])

    if isinstance(expr, (Sum, Product, Power)):
        for arg in expr.args:
            if resolve(arg.is_indeterminate) == Boolean(True):
                return arg

    if all(arg.is_constant_value for arg in expr.args) and not isinstance(expr, Constant):
        try:
            return Value(expr.eval())
        except ExpressionEvaluationException:
            pass

    expr = simplify_associative(expr)
    expr = simplify_commutative(expr)
    expr = simplify_idempotent(expr)

    return expr


@functools.cache
def simplify(expr: Expression) -> Expression:
    if resolve(expr.is_indeterminate) == Boolean(True):
        return expr

    expr = simplify_abstract_arg_expr(expr)

    if isinstance(expr, Sum):
        # combine factors
        args: dict[Expression, list[Expression]] = defaultdict(list)

        for arg in expr.args:
            if isinstance(arg, Product):
                numbers = list(filter(lambda x: isinstance(x, Value), arg.args))
                others = list(filter(lambda x: not isinstance(x, Value), arg.args))

                if not numbers:
                    numbers = [Value(1)]
                if not others:
                    others = [Value(1)]

                args[Product.from_args(*others)].append(Product.from_args(*numbers))
            else:
                args[arg].append(Value(1))

        return simplify_abstract_arg_expr(
            Sum.from_args(*(
                simplify(Product.from_args(key, Sum.from_args(*value))) for key, value in args.items()
            ))
        )

    elif isinstance(expr, Product):
        # combine powers
        args: dict[Expression, list[Expression]] = defaultdict(list)

        for arg in expr.args:
            if arg == Value(0):
                return Value(0)

            if isinstance(arg, Power):
                base_args = arg.base
                if isinstance(base_args, Product):
                    for base_factor in base_args.args:
                        args[base_factor].append(arg.exponent)
                else:
                    args[arg.base].append(arg.exponent)
            else:
                args[arg].append(Value(1))

        return simplify_abstract_arg_expr(
            Product.from_args(*(
                simplify(Power.from_args(key, Sum.from_args(*value))) for key, value in args.items()
            ))
        )

    elif isinstance(expr, Power):
        if isinstance(expr.base, Power):
            return simplify(Power.from_args(expr.base.base, Product.from_args(expr.base.exponent, expr.exponent)))

        # if expr.base == Number(0):
        #     return Number(0)

        if expr.exponent == Value(1):
            return expr.base

        if expr.exponent == Value(0) and expr.base != Value(0):
            return Value(1)

        if isinstance(expr.base, Product):
            return simplify(Product.from_args(*(Power.from_args(arg, expr.exponent) for arg in expr.base.args)))

        return expr

    elif isinstance(expr, Equality):
        if expr.lhs == expr.rhs:
            return Boolean(True)
        if simplify(LogicalOr.from_args(expr.lhs.is_indeterminate, expr.rhs.is_indeterminate)) == Boolean(True):
            return Boolean(False)

    elif isinstance(expr, Logarithm):
        if expr.base == expr.argument:
            return Value(1)

        if expr.argument == Value(1):
            return Value(0)

    elif isinstance(expr, Piecewise):
        if Boolean(False) in expr.args[1::2]:
            return simplify(expr.from_args(*itertools.chain(
                *filter(lambda arg_cond: arg_cond[1] != Boolean(False), expr.iter_args())
            )))

        if len(expr.args) == 2 and expr.args[1] == Boolean(True):
            return simplify(expr.args[0])

    elif isinstance(expr, LogicalOr):
        if Boolean(True) in expr.args:
            return Boolean(True)

    elif isinstance(expr, LogicalNot):
        if isinstance(expr.args[0], LogicalNot):
            return expr.args[0].args[0]

    return expr


def differentiate(expr: Expression, var: Variable) -> Expression:
    if expr.is_constant_value:
        return Value(0)

    if isinstance(expr, Variable):
        if expr == var:
            return Value(1)
        else:
            return Value(0)

    if expr.args:
        # see https://en.wikipedia.org/wiki/Chain_rule#Case_of_scalar-valued_functions_with_multiple_inputs

        args = []

        for i, arg in enumerate(expr.args):
            args.append(
                Product.from_args(
                    differentiate(arg, var),
                    expr.par_diff(i, arg)
                )
            )

        return Sum.from_args(*args)

    raise NotImplementedError(f"Can't differentiate {expr} with respect to {var}.")


def create_func(name: str, function: typing.Callable[[Expression, ...], Expression]) -> Expression:
    def doit(self):
        return function(*(super(self.__class__, self).doit()).args)

    def raise_not_implemented(self: Expression) -> float:
        raise NotImplementedError(f"Can't evaluate {self}.")

    # noinspection PyTypeChecker
    return type(name, (FunctionSyntaxExpression,), {
        "doit": doit,
        "eval": raise_not_implemented
    })


def substitute(expr: Expression, subst_dict: dict[Expression, Expression]):
    if expr in subst_dict:
        return subst_dict[expr]

    args = []

    for arg in expr.args:
        args.append(substitute(arg, subst_dict))

    return expr.from_args(*args)


# def substitute2(expr: Expression, func: typing.Callable[[Expression], Expression | None]):
#     if out := func(expr) is not None:
#         return out
#
#     args = []
#
#     for arg in expr.args:
#         args.append(substitute2(arg, func))
#
#     return expr.from_args(*args)


def reduce(expr: Expression, variable: Expression, depth: int = 10) -> Expression:
    if depth <= 0:
        return expr

    expr = simplify(expr)

    if not isinstance(variable, Variable):
        vars_in_expr = set(filter(lambda x: isinstance(x, Variable), expr.walk()))

        vars_sol = {var: reduce(expr, var, depth=depth - 1) for var in vars_in_expr}

        out = substitute(variable, {var: sol.rhs for var, sol in vars_sol.items()})

        return simplify(Equality.from_args(variable, out))

    if isinstance(expr, Equality):
        if variable in expr.rhs:
            new_eq1 = simplify(expr.from_args(Difference.from_args(expr.lhs, expr.rhs), Value(0)))
            # new_eq2 = simplify(expr.from_args(Quotient.from_args(expr.lhs, expr.rhs), Value(1)))
            new_eq2 = Value(False)
            return reduce(LogicalOr.from_args(new_eq1, new_eq2), variable, depth=depth - 1)

        out = []
        for i, arg in enumerate(expr.lhs.args):
            if variable in arg:
                arg_sols = expr.lhs.inverted(i, expr.rhs)

                u = UniqueVariable()

                var_sol = reduce(Equality(arg, u), variable, depth=depth - 1)

                out += (
                    substitute(var_sol, {u: arg_sol})
                    for arg_sol in arg_sols
                )

        if not out:
            return expr

        return simplify(LogicalOr.from_args(*out))

    elif isinstance(expr, LogicalOr):
        return simplify(LogicalOr.from_args(*(
            reduce(arg, variable, depth=depth - 1) for arg in expr.args
        )))

    elif isinstance(expr, LogicalNot):
        return simplify(LogicalNot(reduce(expr.args[0], variable, depth=depth - 1)))

    elif isinstance(expr, Value):
        return expr

    return expr


def resolve(expr: Expression, *assumptions: Expression) -> Expression:
    # TODO: Implement
    if isinstance(expr, Value):
        return expr

    return simplify(expr)


def _subst(expr: Expression, *args: Expression):
    return substitute(expr, dict(zip(args[::2], args[1::2])))


class _Out(FunctionSyntaxExpression):
    min_args = 1
    max_args = 1


class _NthArg(FunctionSyntaxExpression):
    min_args = 2
    max_args = 2

    def doit(self) -> Expression:
        try:
            return self.args[0].args[int(self.args[1].eval())]
        except (ExpressionEvaluationException, ValueError, IndexError):
            return self


_Differentiate = create_func("D", differentiate)
_Reduce = create_func("Reduce", reduce)
_Substitute = create_func("Subst", _subst)
_NthArg = create_func("NthArg", _NthArg.from_args)
_Out = create_func("Out", _Out.from_args)


def gen_fuzz(length=20):
    possibilities = string.ascii_letters + "*+-/^" * 20 + " " * 8 + string.digits * 2
    return "".join([random.choice(possibilities) for _ in range(length)])


def interactive_fuzz():
    rec = None

    while 1:
        fuzz = gen_fuzz()

        try:
            expr = Expression.from_str(fuzz)
            # print(f" > {fuzz}")
            # print(f" = {expr}")
            a = expr  # .simplify()
            # print(f" = {a}")

            if (rec is None) or len(rec) < len(str(a)):
                rec = str(a)
                print(f" -> {rec}")
            # break
        except (ParsingException, ZeroDivisionError) as e:
            # print(" - ERR", e)
            ...


def generate_random_expression(max_levels: 5) -> Expression:
    if max_levels == 0:
        if random.random() > .5:
            return Value((random.random() * 100 - 50) // 1)
        else:
            return Variable(random.choice("x"))

    expr_type: type[_OperatorExpression] = random.choice((Sum, Product, Power))

    if expr_type.max_args:
        args = [generate_random_expression(max_levels - 1) for _ in range(random.randint(2, expr_type.max_args))]
    else:
        args = [generate_random_expression(max_levels - 1) for _ in range(random.randint(2, 3))]

    return expr_type.from_args(*args)


def main2():
    i = 0
    cells: dict[_Out, Expression] = {}
    while 1:
        cell_num = str(i)

        expr = input(f" [{cell_num}] ")
        if not expr.strip():
            # expr = generate_random_expression(3).to_str_formal()
            # expr = f"Reduce(D(D({expr}, x), x)=0, x)"
            continue
        try:
            expr = Expression.from_str(expr)
        except ParsingException as e:
            print(f" -> {e}")
            continue

        expr = substitute(expr, {Variable("e"): Constant.e, Variable("pi"): Constant.pi, **cells})

        expr = expr.doit()

        # print(f" {' '* len(cell_num)}>  {expr}")
        # print(f" {' ' * len(cell_num)}>  {expr.to_str_full()}")

        simple = simplify(expr)

        # print(f" {' ' * len(cell_num)}>  {simple}")
        print(f" {' ' * len(cell_num)}>  {simple.to_str_human()}")
        if resolve(simple.is_indeterminate) == Boolean(True):
            print(f" {' ' * len(cell_num)}!  Expression is undefined.")

        cells[_Out(Value(i))] = simple

        print()
        i += 1


if __name__ == '__main__':
    main2()
