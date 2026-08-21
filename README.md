# Pento chess solver

Pento chess fills an 8x8 board using twelve pentominoes and one tetromino.

Naively trying 64 positions, 4 rotations, and 2 flips for each piece gives a loose upper bound of `(64 * 4 * 2) ** 13 ≈ 1.66e35` placement combinations. Even after removing out-of-bounds and duplicate placements, about `1.73e28` combinations remain before overlap pruning.

The solver uses depth-first search with a minimum-remaining-values heuristic. At each step, it chooses the smaller candidate set from:

1. Legal placements for the unused piece with the fewest legal placements.
2. Legal placements covering the uncovered cell with the fewest covering placements.

If there is a piece that can not be placed or a cell that can not be filled, the path is exhausted.

Placements are precomputed, grouped by piece, and indexed by covered cell. Board occupancy and remaining pieces are represented by 64-bit and 13-bit masks, making overlap and availability checks fast.
