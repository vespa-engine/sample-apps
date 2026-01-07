# Closeness Function Misleading for Binary Vectors with Hamming Distance

## The Problem

Vespa's built-in `closeness(field, embedding)` function computes:

```
closeness = 1 / (1 + distance)
```

This works well for distance metrics like euclidean or angular where similar vectors have distances close to 0. However, for **binary vectors with hamming distance**, this formula produces misleadingly small values.

### Example

For 768-bit binary vectors (96-dim int8 with `pack_bits`):
- Two semantically similar documents might have hamming distance ~250 (differing in ~33% of bits)
- `closeness = 1 / (1 + 250) = 0.00398`

This value appears to indicate poor similarity, but it's actually a good match. The closeness values will **never approach 1** unless vectors are nearly identical bit-for-bit.

### Query Results Comparison

| Metric | Best Match | Worst Match | Intuitive? |
|--------|------------|-------------|------------|
| closeness | 0.00395 | 0.00314 | No |
| similarity | 0.672 | 0.587 | Yes |

## Solution: Normalized Binary Closeness

Use a normalized similarity function instead:

```
similarity = 1 - (hamming_distance / max_hamming_distance)
```

For 96-dim int8 vectors: `max_hamming_distance = 96 * 8 = 768`

```
function similarity() {
    expression: 1 - (distance(field, embedding) / 768)
}
```

This produces values in the intuitive [0, 1] range where:
- 1.0 = identical vectors
- 0.0 = maximally different vectors
- 0.67 = 67% of bits match

## Documentation Updates Needed

1. **Binarizing vectors docs**: Add note that `closeness()` produces unintuitive values for hamming distance; recommend using normalized similarity instead.

2. **Closeness function docs**: Add section explaining that closeness is designed for distance metrics where similar items have distance → 0, and provide the normalized alternative for hamming distance.

3. **Nearest neighbor docs**: When showing hamming distance examples, use `distance()` with normalization rather than `closeness()` in ranking examples.
