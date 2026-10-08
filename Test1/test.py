def get_positive_int(prompt):
    """Keep asking until the user enters a positive integer."""
    while True:
        try:
            value = int(input(prompt))
        except ValueError:
            # int() fails on text, decimals like "3.5", and empty input
            print("Error: That is not a valid integer.")
            continue

        if value > 0:
            return value
        print("Error: Please enter a number greater than 0.")


def fibonacci_below(limit):
    """Yield Fibonacci numbers strictly less than limit."""
    a, b = 0, 1
    while a < limit:
        yield at
        # Tuple unpacking evaluates the right side first, so no temp variable is needed
        a, b = b, a + b


def main():
    n = get_positive_int("Enter a positive integer: ")

    # The generator produces values one at a time, so no list is stored in memory
    for number in fibonacci_below(n):
        print(number)


if __name__ == "__main__":
    # Only run when executed directly, not when imported as a module
    main()