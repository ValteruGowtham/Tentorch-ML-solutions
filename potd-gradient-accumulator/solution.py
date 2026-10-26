def gradient_accumulator(
    theta: float, eta: float, microbatches: list[tuple[int, float]]
) -> tuple[float, float]:
    total_examples = sum(n for n, _ in microbatches)
    total_gradient = sum(n * g for n, g in microbatches)

    g = total_gradient / total_examples
    theta_prime = theta - eta * g

    return round(g, 6), round(theta_prime, 6)
