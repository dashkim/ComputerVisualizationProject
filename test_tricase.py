"""
test_tricase.py

Purpose:
    This script checks whether a Marching Cubes triCase[256][16] table is
    consistent with our class's vertex and edge numbering convention.

How to run:
    python3 test_tricase.py tricase.cxx

What it verifies:
    For each case number 0–255, the binary representation of the case number
    determines which cube vertices are marked/inside.

    bit 0 -> V0
    bit 1 -> V1
    bit 2 -> V2
    bit 3 -> V3
    bit 4 -> V4
    bit 5 -> V5
    bit 6 -> V6
    bit 7 -> V7

    An edge should appear in triCase[case] only if its two endpoints have
    different bit values. That means the isosurface crosses that edge.

    Example:
        If E1 = V1-V3, then E1 is valid for a case only when V1 and V3
        have opposite values, i.e. one is marked and one is unmarked.

Our class edge convention:
    E0  = V0-V1
    E1  = V1-V3
    E2  = V2-V3
    E3  = V0-V2

    E4  = V4-V5
    E5  = V5-V7
    E6  = V6-V7
    E7  = V4-V6

    E8  = V0-V4
    E9  = V1-V5
    E10 = V2-V6
    E11 = V3-V7

Input format:
    The input file should contain the C/C++ table rows in this style:

        {0, 3, 8, -1, -1, ...},  /* 1 */

    The comment after each row must contain the decimal case number.
    The script uses that comment to identify which case the row belongs to.

Optional filtering:
    By default, the script checks all 256 cases.

    To check only cases with exactly k marked vertices, uncomment the filter
    inside the main loop and replace k with the desired number:

        if bin(case_num).count("1") != k:
            continue

    Example:
        k = 3 checks only the 3-vertex cases.

What an error means:
    "uses E#, but E#=Va-Vb does not cross"
        The row contains an edge whose endpoints are both marked or both
        unmarked. That edge should not appear for that case.

    "missing crossed edges"
        There is an edge whose endpoints differ, so the surface should cross it,
        but the row never uses that edge.

    "degenerate triangle"
        A triangle uses the same edge more than once, such as {8, 10, 8}.

Important limitation:
    This script checks edge membership and table structure. It does NOT fully
    prove that triangle winding/order is correct. A row can pass this script
    and still have inconsistent winding or a questionable triangulation.
"""
#!/usr/bin/env python3
import re
import sys

# Your class convention:
# bit i corresponds to Vi
edges = {
    0:  (0, 1),
    1:  (1, 3),
    2:  (2, 3),
    3:  (0, 2),

    4:  (4, 5),
    5:  (5, 7),
    6:  (6, 7),
    7:  (4, 6),

    8:  (0, 4),
    9:  (1, 5),
    10: (2, 6),
    11: (3, 7),
}

def bit(case_num, vertex):
    return (case_num >> vertex) & 1

def expected_crossed_edges(case_num):
    crossed = set()

    for e, (a, b) in edges.items():
        if bit(case_num, a) != bit(case_num, b):
            crossed.add(e)

    return crossed

def parse_rows(filename):
    rows = {}

    with open(filename, "r") as f:
        text = f.read()

    # Match rows like:
    # {0, 3, 8, -1, ...}, /* 1 */
    pattern = re.compile(r"^\s*\{([^{}]*)\}\s*,?\s*/\*\s*(\d+)", re.MULTILINE)

    for match in pattern.finditer(text):
        nums_text = match.group(1)
        case_num = int(match.group(2))

        nums = [int(x.strip()) for x in nums_text.split(",") if x.strip()]
        rows[case_num] = nums

    return rows

def check_row(case_num, row):
    errors = []

    if len(row) != 16:
        errors.append(f"row has {len(row)} entries, expected 16")

    for x in row:
        if x != -1 and not (0 <= x <= 11):
            errors.append(f"invalid edge value {x}")

    # Find first -1
    try:
        stop = row.index(-1)
    except ValueError:
        errors.append("row has no -1 terminator")
        stop = len(row)

    used = row[:stop]

    # After first -1, everything should also be -1
    for i in range(stop, len(row)):
        if row[i] != -1:
            errors.append(f"value after -1 at index {i}: {row[i]}")

    if len(used) % 3 != 0:
        errors.append(f"number of used edge entries is {len(used)}, not divisible by 3")

    # No repeated edge inside a single triangle
    for i in range(0, len(used), 3):
        tri = used[i:i+3]
        if len(tri) == 3 and len(set(tri)) != 3:
            errors.append(f"degenerate triangle {tri}")

    expected = expected_crossed_edges(case_num)
    used_set = set(used)

    # Every used edge must be a crossed edge
    for e in used_set:
        if e not in expected:
            a, b = edges[e]
            errors.append(
                f"uses E{e}, but E{e}=V{a}-V{b} does not cross "
                f"({bit(case_num, a)}-{bit(case_num, b)})"
            )

    # Every crossed edge should usually appear at least once,
    # unless the case is intentionally empty.
    missing = expected - used_set
    if missing:
        errors.append(f"missing crossed edges: {sorted(missing)}")

    return errors

def main():
    if len(sys.argv) != 2:
        print("Usage: ./test_tricase.py tricase.cxx")
        sys.exit(1)

    filename = sys.argv[1]
    rows = parse_rows(filename)

    all_errors = False
    wrong_count = 0
    checked_count = 0

    if len(rows) != 256:
        print(f"ERROR: found {len(rows)} rows, expected 256")
        all_errors = True

    for case_num in range(256):
        # Filter by case ie != 4 would skip all non 4-vertex cases
        #if bin(case_num).count("1") != 4:
           # continue

        if case_num not in rows:
            print(f"ERROR case {case_num}: missing row")
            all_errors = True
            wrong_count += 1
            continue

        checked_count += 1

        errors = check_row(case_num, rows[case_num])
        if errors:
            all_errors = True
            wrong_count += 1
            print(f"\nCase {case_num} = {case_num:08b}")
            print(f"row: {rows[case_num]}")
            print(f"expected crossed edges: {sorted(expected_crossed_edges(case_num))}")
            for err in errors:
                print(f"  - {err}")

    print()
    print(f"Checked rows: {checked_count}")
    print(f"Wrong rows:   {wrong_count}")
    print(f"Correct rows: {checked_count - wrong_count}")

    if not all_errors:
        print("All rows passed the edge-crossing convention check.")
    else:
        print("\nFinished with errors.")
        sys.exit(1)

if __name__ == "__main__":
    main()
