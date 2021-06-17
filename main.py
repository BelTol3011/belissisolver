def parse_util(term: str, split_char: str) -> list[str]:
    groups: list[str] = [""]
    level = 0
    for char in term:
        if char == "(":
            level += 1
        elif char == ")":
            level -= 1
        elif char == split_char and level == 0:
            groups.append("")
        else:
            groups[-1] += char

    return groups


class MathElement:
    def evaluate(self, **variables: dict[str: float]) -> float:
        raise NotImplementedError()

    @classmethod
    def from_str(cls, term: str) -> "MathElement":
        # find sums
        factors = parse_util(term, "+")

        if len(factors) == 1:
            sums = parse_util(factors[0], "*")
            if len(sums) == 1:
                # this is a number
                return Number(sums[0])

            # this is a product
            return Product([MathElement.from_str(part_term) for part_term in sums])
        else:
            return Sum([MathElement.from_str(part_term) for part_term in factors])


class Number(MathElement):
    def __init__(self, value: str):
        self.value = value.lstrip().rstrip()

    def evaluate(self, **variables: dict[str: float]) -> float:
        if self.value in variables:
            return variables[self.value]
        else:
            return float(self.value)

    def __str__(self):
        return "$" + str(self.value)


class Sum(MathElement):
    def __init__(self, summands):
        self.summands: list[MathElement] = summands

    def evaluate(self, **variables: dict[str: float]) -> float:
        return sum([summand.evaluate(**variables) for summand in self.summands])

    def __str__(self):
        return "(" + ") + (".join([str(summand) for summand in self.summands]) + ")"


class Product(MathElement):
    def __init__(self, factors):
        self.factors: list[MathElement] = factors

    def evaluate(self, **variables: dict[str: float]) -> float:
        # there is no prod() function so this is no list comprehension :((
        product = 1
        for factor in self.factors:
            product *= factor.evaluate(**variables)

        return product

    def __str__(self):
        return "(" + ") * (".join([str(factor) for factor in self.factors]) + ")"


def main():
    while 1:
        inp = input(" ~ ")
        try:
            math_expr = MathElement.from_str(inp)
        except Exception as e:
            print(e)
            continue
        print(math_expr)
        print(" = ", end="")
        try:
            print(math_expr.evaluate())
        except ValueError as e:
            print(f"? ({e})")


if __name__ == '__main__':
    main()
