def get_optimal_denominations(amount):
  denominations = [500, 200, 100, 50, 20, 10, 5, 2, 1]
  result = {}

  for d in denominations:
    count = amount // d 
    if count > 0:
      result[d] = count
      amount = amount % d

return result 
