def pass_at_1(results):
    n = len(results)
    k = sum(1 for r in results if r.get("passed"))
    return 0.0 if n == 0 else k / n