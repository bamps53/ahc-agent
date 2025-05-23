"""
Sample problem for testing AHCAgent CLI.
"""

# AtCoder Heuristic Contest Sample Problem

## Problem Statement

You are given a grid of size N×N. Each cell (i, j) has a value A[i][j].

Your task is to select exactly K cells from the grid such that the sum of the values of the selected cells is maximized.

## Input Format

The first line contains two integers N and K (1 ≤ N ≤ 100, 1 ≤ K ≤ N×N).
The next N lines each contain N integers. The j-th integer in the i-th line represents A[i][j] (0 ≤ A[i][j] ≤ 1000).

## Output Format

Output K lines, each containing two integers i and j (1 ≤ i, j ≤ N), representing the coordinates of the selected cells.
Each cell must be selected at most once.

## Scoring

Your score is the sum of the values of the selected cells.

## Constraints

- 1 ≤ N ≤ 100
- 1 ≤ K ≤ N×N
- 0 ≤ A[i][j] ≤ 1000

## Sample Input 1

```
5 3
10 20 30 40 50
60 70 80 90 100
110 120 130 140 150
160 170 180 190 200
210 220 230 240 250
```

## Sample Output 1

```
5 5
5 4
4 5
```

## Sample Score 1

250 + 240 + 200 = 690
