from collections.abc import Sequence
from time import perf_counter
from typing import NamedTuple

import numpy as np


class Position(NamedTuple):
    """A board coordinate."""

    y: int
    x: int


class Action(NamedTuple):
    """Place one oriented piece at a board position."""

    pi: int
    pos: Position
    rot: int
    flip: int


class Placement(NamedTuple):
    """An action and the board cells it occupies."""

    action: Action
    mask: int


pieces = [
    [
        [1, 0, 1],
        [1, 1, 1],
    ],
    [
        [1, 1, 0],
        [1, 1, 1],
    ],
    [
        [1, 1, 1, 1, 1],
    ],
    [
        [1, 1, 0, 0],
        [0, 1, 1, 1],
    ],
    [
        [0, 0, 1, 0],
        [1, 1, 1, 1],
    ],
    [
        [0, 0, 0, 1],
        [1, 1, 1, 1],
    ],
    [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ],
    [
        [1, 0, 0],
        [1, 0, 0],
        [1, 1, 1],
    ],
    [
        [0, 1, 0],
        [0, 1, 0],
        [1, 1, 1],
    ],
    [
        [0, 1, 1],
        [1, 1, 0],
        [0, 1, 0],
    ],
    [
        [1, 1, 0],
        [0, 1, 0],
        [0, 1, 1],
    ],
    [
        [0, 0, 1],
        [0, 1, 1],
        [1, 1, 0],
    ],
    [
        [1, 1],
        [1, 1],
    ],
]


pieces = [np.array(piece, dtype=bool) for piece in pieces]
orientations = [
    [
        [np.rot90(np.fliplr(piece) if flip else piece, rot) for flip in range(2)]
        for rot in range(4)
    ]
    for piece in pieces
]
possible_actions: list[list[Action]] = [
    [
        Action(pi, Position(posy, posx), rot, flip)
        for posy in range(8)
        for posx in range(8)
        for rot in range(4)
        for flip in range(2)
        if posy + orientations[pi][rot][flip].shape[0] <= 8
        and posx + orientations[pi][rot][flip].shape[1] <= 8
    ]
    for pi in range(len(pieces))
]
orientation_cells: list[list[list[tuple[Position, ...]]]] = [
    [
        [
            tuple(Position(int(y), int(x)) for y, x in np.argwhere(piece))
            for piece in flips
        ]
        for flips in rotations
    ]
    for rotations in orientations
]
action_cells: dict[Action, tuple[Position, ...]] = {
    action: tuple(
        Position(action.pos.y + cell.y, action.pos.x + cell.x)
        for cell in orientation_cells[action.pi][action.rot][action.flip]
    )
    for actions in possible_actions
    for action in actions
}
action_masks: dict[Action, int] = {
    action: sum(1 << (63 - cell.y * 8 - cell.x) for cell in cells)
    for action, cells in action_cells.items()
}
npieces = len(pieces)
full_board = (1 << 64) - 1
all_pieces = (1 << npieces) - 1


def unique_placements(piece_index: int) -> tuple[Placement, ...]:
    """Return one action for each distinct placement of a piece."""
    placements: dict[int, Placement] = {}
    for action in possible_actions[piece_index]:
        mask = action_masks[action]
        placements.setdefault(mask, Placement(action, mask))
    return tuple(placements.values())


possible_placements: tuple[tuple[Placement, ...], ...] = tuple(
    unique_placements(pi) for pi in range(npieces)
)
placements_by_cell: list[list[Placement]] = [[] for _ in range(64)]
for piece_placements in possible_placements:
    for placement in piece_placements:
        bits = placement.mask
        while bits:
            bit = bits & -bits
            placements_by_cell[bit.bit_length() - 1].append(placement)
            bits ^= bit

actions: list[Action] = [
    Action(
        0,
        Position(0, 6),
        1,
        0,
    ),
    # Action(
    #     1,
    #     Position(6, 1),
    #     0,
    #     0,
    # ),
    # Action(
    #     2,
    #     Position(3, 7),
    #     1,
    #     0,
    # ),
    # Action(
    #     3,
    #     Position(0, 1),
    #     2,
    #     1,
    # ),
    # Action(
    #     4,
    #     Position(4, 5),
    #     1,
    #     1,
    # ),
    # Action(
    #     6,
    #     Position(0, 4),
    #     0,
    #     0,
    # ),
    # Action(
    #     9,
    #     Position(1, 2),
    #     1,
    #     1,
    # ),
    # Action(
    #     10,
    #     Position(3, 4),
    #     0,
    #     1,
    # ),
    # Action(
    #     11,
    #     Position(5, 3),
    #     0,
    #     1,
    # ),
]


def state_from_actions(starting_actions: Sequence[Action]) -> tuple[int, int]:
    """Validate actions and return their occupied and remaining masks."""
    occupied = 0
    remaining = all_pieces
    for action in starting_actions:
        piece_index = action.pi
        piece_bit = 1 << piece_index
        mask = action_masks.get(action)
        if mask is None or not remaining & piece_bit or occupied & mask:
            raise ValueError(f"invalid action: {action}")
        occupied |= mask
        remaining ^= piece_bit
    return occupied, remaining


def select_placements(occupied: int, remaining: int) -> list[Placement]:
    """Return legal rows for the most constrained exact-cover column."""
    best: list[Placement] = []

    # actions
    for pi, pls in enumerate(possible_placements):
        if remaining & (1 << pi):
            legal = [pl for pl in pls if not (pl.mask & occupied)]
            if not legal:
                return []
            if not best or len(legal) < len(best):
                best = legal

    # cells
    uncovered = full_board ^ occupied
    while uncovered:
        bit = uncovered & -uncovered
        placements = placements_by_cell[bit.bit_length() - 1]
        legal = [
            pl
            for pl in placements
            if remaining & (1 << pl.action.pi) and not (pl.mask & occupied)
        ]
        if not legal:
            return []
        if not best or len(legal) < len(best):
            best = legal
        uncovered ^= bit
    return best


def search(occupied: int, remaining: int, path: list[Action]) -> list[Action] | None:
    """Find an exact cover from the current bitmask state."""
    if not remaining:
        return path if occupied == full_board else None
    for placement in select_placements(occupied, remaining):
        result = search(
            occupied | placement.mask,
            remaining ^ (1 << placement.action.pi),
            path + [placement.action],
        )
        if result is not None:
            return result
    return None


def solve(starting_actions: Sequence[Action] = ()) -> list[Action] | None:
    """Solve the board while retaining the supplied starting placements."""
    occupied, remaining = state_from_actions(starting_actions)
    return search(occupied, remaining, list(starting_actions))


def print_board(board_actions: Sequence[Action]) -> None:
    """Print actions as an 8x8 board of hexadecimal piece IDs."""
    state_from_actions(board_actions)
    board: list[list[str]] = [["." for _ in range(8)] for _ in range(8)]
    for action in board_actions:
        piece_id = format(action[0], "X")
        for y, x in action_cells[action]:
            board[y][x] = piece_id
    print("\n".join(" ".join(row) for row in board))


if __name__ == "__main__":
    started = perf_counter()
    print_board(actions)
    solution = solve(actions)
    print_board(solution)
    print(
        f"Solved from {len(actions)} starting pieces in {perf_counter() - started:.6f}s"
    )
