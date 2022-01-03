# A hashiwokakero solver

This program is a solver for the
[Hashiwokakero](https://en.wikipedia.org/wiki/Hashiwokakero) puzzle
(aka "Hashi" or "Bridges").

## How it works

The main strategy involved is to repeatedly apply constraints: At the
most basic level, knowing that a particular island has a given valency
and number of neighbours means certain bridges must exist. For
example, if an island has 3 bridges and 2 neighbours, with a maximum
of 2 bridges between islands, the pigeonhole principle means there
must be at least one bridge to each neighbour.

Adding bridges adds constraints - by blocking off further routes, and
further limiting the valence at the other end of an island-island
route, limiting the number of potential bridges. We can repeatedly
apply constraints to move towards a solution.

A secondary constraint is that the final network must be connected. We
cannot create disconnected components. This constraint is applied
during the other strategy: Case splits. Effectively, when we can't
make progress with the main strategy, we guess connections and discard
those that don't work. Creating disconnected components is a "doesn't
work" case, as well as not having any valid moves remaining.

# How to use it

The input format is as follows:

 * The input must be a rectangular grid.
 * The numbers 1-8 represent islands with that much valence.
 * Empty cells are represented as with a period ('.').

Comments, starting with '#', through to the end of a line, are
ignored. Empty lines are ignored. Leading and trailing whitespace are
ignored.

For example, input can like this:

`
# This is a very, very simple Hashi puzzle.

   .....
   .2.1.
   .....
   .1... # Why am I putting a comment here? Because I can.
   .....

# Goodbye.
`

# Performance

While not deliberately pessimised, using the most efficient algorithms
is a non-goal. Instead, the aim is to keep the code simple, and
minimise the number of invariants that must be maintained. For
example, we scan the full grid each pass to check if it's solved,
rather than track the current state through a union find algorithm or
set of currently active nodes or whatever.

# Development notes

Current plan:

 * -Define data structures-
 * -Build parser/printer-
 * -Add a really basic command-line-
 * -Add code to constrain locally-
 * -Add code to propagate constraints-
 * -Add code to draw in bridges-
 * -Add code to perform case splits-
 * -Add code to check connectedness-
 * -Add some CLI tests-
 * Finalise README.md

Use a fairly TDD approach throughout, use anyhow and clap crates again.
