# Retail-Saarthi
RetailSaarthi: Our semester-end project. An LLM-powered platform designed to help small &amp; medium retailers with predictive inventory analytics and a P2P marketplace for surplus stock. This project explores a secure, AI-driven solution to enhance efficiency and financial resilience in local commerce.

# Denomination Split (Rule-Based)

This module breaks a given cash amount into optimal denominations.

### Usage

```python
from models.denomination_split.engine import split_amount

print(split_amount(1260))
