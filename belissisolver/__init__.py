from __future__ import annotations

import abc
import random
import string
import time
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
    @abc.abstractmethod
    def to_str(self) -> str:
        ...

    def to_pretty_str(self) -> list[str]:
        return [self.to_str()]

    @abc.abstractmethod
    def eval(self) -> float:
        ...

    @classmethod
    @abc.abstractmethod
    def parse(cls, expr: str) -> "Expression":
        ...

    @staticmethod
    def from_str(expr: str) -> "Expression":
        expr = remove_top_level_parens(expr)

        errors = []
        for expr_type in Equality, Sum, Difference, Product, Quotient, Power, Number, Boolean, Variable:
            try:
                return expr_type.parse(expr)
            except (DirectParsingException, ParsingException) as e:
                errors.append(e)

        errors.sort(key=lambda error: isinstance(error, DirectParsingException))

        raise ParsingException(f"Could not parse expression {expr!r}.\n" +
                               "".join(f" - {error_msg}\n" for error_msg in errors))

    @property
    @abc.abstractmethod
    def is_constant(self) -> bool:
        ...

    def __str__(self):
        return self.to_str()

    def __repr__(self):
        return str(self)

    @abc.abstractmethod
    def __hash__(self):
        ...

    @property
    def is_undefined(self):
        return False


class Variable(Expression):
    forbidden_chars = "^*+-/(). "

    def __init__(self, symbol: str):
        if any([char in self.forbidden_chars for char in symbol]):
            raise ParsingException(f"Symbol contains invalid chars: {symbol!r}.")

        if not symbol:
            raise ParsingException("Can't create an empty Symbol.")

        self.symbol = symbol

    @classmethod
    def parse(cls, expr: str) -> "Expression":
        return Variable(expr)

    def eval(self) -> float:
        raise ExpressionEvaluationException(f"Can't evaluate unresolved variable {self.symbol!r}.")

    def to_str(self) -> str:
        return self.symbol

    @property
    def is_constant(self):
        return False

    def __eq__(self, other):
        if not isinstance(other, Variable):
            return False

        return self.symbol == other.symbol

    def __hash__(self):
        return hash(self.symbol)


class Number(Expression):
    def __init__(self, value: float):
        self.value = value

    @classmethod
    def parse(cls, expr: str) -> "Expression":
        try:
            return Number(float(expr))
        except ValueError as e:
            raise DirectParsingException(f"Can't parse {expr!r} as a number.") from e

    def eval(self) -> float:
        return self.value

    def to_str(self) -> str:
        return f"{self.value}"

    @property
    def is_constant(self):
        return True

    def __eq__(self, other):
        if not isinstance(other, Number):
            return False

        return self.value == other.value

    def __hash__(self):
        return hash(self.value)


class Boolean(Number):
    @classmethod
    def parse(cls, expr: str) -> "Expression":
        if expr == "True":
            return Boolean(True)
        elif expr == "False":
            return Boolean(False)

        raise DirectParsingException(f"Can't parse {expr!r} as a boolean.")


T2 = typing.TypeVar("T2", bound=Expression)


class AbstractArgumentExpression(Expression, abc.ABC):
    operator: str
    max_args: int | None = None

    is_commutative: bool
    is_associative: bool
    unit_element: Expression | None = None

    def __init__(self, *args: Expression):
        self.args = args

    @classmethod
    def from_args(cls: typing.Type[T2], *args: Expression) -> T2:
        if len(args) == 1:
            return args[0]

        return cls(*args)

    @classmethod
    def parse(cls, expr: str) -> "Expression":
        str_args = parens_aware_split(expr, cls.operator)

        if cls.max_args and len(str_args) > cls.max_args:
            raise DirectParsingException(f"Could not parse {expr!r} as {cls.__name__}.")

        if len(str_args) > 1:
            return cls.from_args(*(Expression.from_str(arg) for arg in str_args))
        else:
            raise DirectParsingException(f"Can't parse {expr!r} as {cls.__name__}.")

    def to_str(self) -> str:
        return "(" + f" {self.operator} ".join([expr.to_str() for expr in self.args]) + ")"

    @property
    def is_constant(self):
        return all(arg.is_constant for arg in self.args)

    def __hash__(self):
        return hash(tuple(self.args))

    def __contains__(self, item):
        return (item in self.args) or any(item in arg for arg in self.args)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False

        if len(self.args) != len(other.args):
            return False

        if self.is_commutative:
            if set(self.args) == set(other.args):
                return True
        else:
            return all((arg1 == arg2) for arg1, arg2 in zip(self.args, other.args))


class Sum(AbstractArgumentExpression):
    operator = "+"
    unit_element = Number(0)
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

    def eval(self) -> float:
        return sum([arg.eval() for arg in self.args])


class Product(AbstractArgumentExpression):
    operator = "*"
    unit_element = Number(1)
    is_commutative = True
    is_associative = True

    def eval(self) -> float:
        product = 1
        for arg in self.args:
            product *= arg.eval()
        return product


class QuotientsDifferencesMixin:
    def __init__(self, *args: Expression):
        raise ParsingException(
            f"{self.__class__.__name__}s are not meant to be instantiated. Use {self.__class__.__name__}.from_args() "
            f"to yield an equivalent Expression."
        )


class Quotient(AbstractArgumentExpression, QuotientsDifferencesMixin):
    operator = "/"

    @classmethod
    def from_args(cls, *args: Expression) -> Expression:
        assert len(args) >= 2

        return Product(args[0], *[Power(arg, Number(-1)) for arg in args[1:]])

    def eval(self) -> float:
        ...


class Difference(AbstractArgumentExpression):
    operator = "-"

    @classmethod
    def from_args(cls, *args: Expression) -> Sum:
        assert len(args) >= 2

        return Sum(args[0], *[Product(arg, Number(-1)) for arg in args[1:]])

    def eval(self) -> float:
        ...


class Power(AbstractArgumentExpression):
    operator = "^"
    max_args = 2
    is_commutative = False
    is_associative = False

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
    def is_undefined(self):
        if self.exponent == Number(0) and self.base == Number(0):
            return True

        # noinspection PyUnresolvedReferences
        if isinstance(self.exponent, Number) and self.exponent.value < 0 and self.base == Number(0):
            return True


class Equality(AbstractArgumentExpression):
    operator = "="
    max_args = 3
    is_commutative = True
    is_associative = False

    def eval(self) -> float:
        return self.args[0].eval() == self.args[1].eval()


T = typing.TypeVar("T", bound=Expression)


def simplify_associative(expr: T) -> T:
    if not isinstance(expr, AbstractArgumentExpression) or not expr.is_associative:
        return expr

    args = []

    for arg in expr.args:
        if isinstance(arg, expr.__class__):
            args += simplify_associative(arg).args
        else:
            args.append(arg)

    return expr.__class__(*args)


def simplify_commutative(expr: T) -> T:
    if not isinstance(expr, AbstractArgumentExpression) or not expr.is_commutative:
        return expr

    args: list[Expression] = []
    constants = []
    numbers = []

    assert expr.unit_element is not None

    for arg in expr.args:
        if arg != expr.unit_element:
            if isinstance(arg, Number):
                numbers.append(arg)
            elif arg.is_constant:
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


def simplify_no_recurse(expr: T) -> T:
    if isinstance(expr, Number):
        return expr

    if isinstance(expr, AbstractArgumentExpression):
        for arg in expr.args:
            if arg.is_undefined:
                return arg
        else:
            expr = expr.__class__.from_args(*[simplify(arg) for arg in expr.args])

        if all(arg.is_constant for arg in expr.args):
            try:
                return Number(expr.eval())
            except ExpressionEvaluationException:
                pass

        expr = simplify_associative(expr)
        expr = simplify_commutative(expr)

    return expr


def simplify(expr: Expression) -> Expression:
    expr = simplify_no_recurse(expr)

    if isinstance(expr, Sum):
        # combine factors
        args: dict[Expression, list[Expression]] = defaultdict(list)

        for arg in expr.args:
            if isinstance(arg, Product):
                numbers = filter(lambda x: isinstance(x, Number), arg.args)
                others = filter(lambda x: not isinstance(x, Number), arg.args)

                args[Product.from_args(*others)].append(Product.from_args(*numbers))
            else:
                args[arg].append(Number(1))

        return simplify_no_recurse(
            Sum.from_args(*(
                simplify(Product.from_args(key, Sum.from_args(*value))) for key, value in args.items()
            ))
        )

    elif isinstance(expr, Product):
        # combine powers
        args: dict[Expression, list[Expression]] = defaultdict(list)

        for arg in expr.args:
            if arg == Number(0):
                return Number(0)

            if isinstance(arg, Power):
                base_args = arg.base
                if isinstance(base_args, Product):
                    for base_factor in base_args.args:
                        args[base_factor].append(arg.exponent)
                else:
                    args[arg.base].append(arg.exponent)
            else:
                args[arg].append(Number(1))

        return simplify_no_recurse(
            Product.from_args(*(
                simplify(Power.from_args(key, Sum.from_args(*value))) for key, value in args.items()
            ))
        )

    elif isinstance(expr, Power):
        if isinstance(expr.base, Power):
            return simplify(Power.from_args(expr.base.base, Product.from_args(expr.base.exponent, expr.exponent)))

        # if expr.base == Number(0):
        #     return Number(0)

        if expr.exponent == Number(1):
            return expr.base

        if expr.exponent == Number(0) and expr.base != Number(0):
            return Number(1)

        return expr

    return expr


def differentiate(expr: Expression, var: Variable) -> Expression:
    if expr.is_constant:
        return Number(0)

    if isinstance(expr, Variable):
        if expr == var:
            return Number(1)
        else:
            return Number(0)

    elif isinstance(expr, Sum):
        return Sum.from_args(*(differentiate(arg, var) for arg in expr.args))

    elif isinstance(expr, Product):
        factors = expr.args
        diff_factors = [differentiate(factor, var) for factor in factors]

        return Sum.from_args(*(
            Product.from_args(diff_factors[i], *factors[:i], *factors[i + 1:])
            for i in range(len(factors))
        ))

    elif isinstance(expr, Power):
        if expr.base == var:
            return Product.from_args(expr.exponent,
                                     Power.from_args(var, Difference.from_args(expr.exponent, Number(1))))

    raise NotImplementedError(f"Can't differentiate {expr} with respect to {var}.")


def substitute(expr: Expression, key: Expression, to: Expression):
    if isinstance(expr, AbstractArgumentExpression):
        args = []

        for arg in expr.args:
            if arg == key:
                args.append(to)
            elif isinstance(arg, AbstractArgumentExpression):
                args.append(substitute(arg, key, to))
            else:
                args.append(arg)

        return expr.from_args(*args)


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
            return Number((random.random() * 100 - 50) // 1)
        else:
            return Variable(random.choice("x"))

    expr_type: type[AbstractArgumentExpression] = random.choice((Sum, Product, Power))

    if expr_type.max_args:
        args = [generate_random_expression(max_levels - 1) for _ in range(random.randint(2, expr_type.max_args))]
    else:
        args = [generate_random_expression(max_levels - 1) for _ in range(random.randint(2, 3))]

    return expr_type.from_args(*args)


def main():
    while 1:
        txt = input("f(x) = ")

        # (((hl * (i ^ -1)) + (-1 * 98t)) + ((S * (O * (Ft ^ -1))) + (-1 * (h * (QB ^ -1)))))
        # (A + ((((EMOH ^ 1.0) * (RC ^ -1)) * d) + (-1 * 6.0) + (-1 * lD) + (-1 * d)))

        try:
            if txt == "":
                expr = generate_random_expression(4)
            else:
                expr = Expression.from_str(txt)
        except ParsingException as e:
            print(f"f(x) = ? ({e})")
            continue

        print(f"f(x) = {expr}")

        t1 = time.perf_counter()
        simple = simplify(expr)
        dt = time.perf_counter() - t1

        if simple != expr:
            print(f"f(x) = {simple}")

        if simple.is_undefined:
            print("// undefined")

        print(f"// simplification took {dt:.4f}s")

        try:
            evaluated = simple.eval()

            if Number(evaluated) != simple:
                print(f"f(x) ≈ {evaluated}")
        except ExpressionEvaluationException as e:
            print(f"f(x) ≈ ? ({e})")

        try:
            t2 = time.perf_counter()
            diff = simplify(differentiate(simple, Variable("x")))
            dt2 = time.perf_counter() - t2

            print(f"f'(x) = {diff}")
            print(f"// differentiation took {dt2:.4f}s")
        except NotImplementedError as e:
            print(f"f'(x) = ? ({e})")
        print()


if __name__ == '__main__':
    main()
