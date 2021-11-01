# A hashiwokakero solver

This program is a solver for the
[Hashiwokakero](https://en.wikipedia.org/wiki/Hashiwokakero) puzzle.

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

TODO

# Development notes

Current plan:

 * Define data structures
 * Build parser/printer
 * Add code to constrain locally
 * Add code to propagate constraints
 * Add code to draw in bridges
 * Add code to perform case splits
 * Add code to check connectedness

Use a fairly TDD approach throughout, use anyhow and clap crates again.
