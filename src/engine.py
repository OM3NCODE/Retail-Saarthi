from .rules import get_available_denominations

def split_amount(amount: int) -> dict:
    result = {}
    remaining = amount

    for denom in get_available_denominations():
        count = remaining // denom
        result[denom] = count
        remaining %= denom

    result["remaining"] = remaining   # in case amount is not divisible
    return result
