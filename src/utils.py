def validate_amount(amount):
    if not isinstance(amount, int):
        raise ValueError("Amount must be an integer.")
    if amount < 0:
        raise ValueError("Amount cannot be negative.")
    return amount
