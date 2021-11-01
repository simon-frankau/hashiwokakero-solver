//
// Hashiwokakero solver
//
// Copyright 2021 Simon Frankau
//

////////////////////////////////////////////////////////////////////////
// Data structures / problem representation
//

// When constraint-solving, we want to know the max/min number of
// bridges that may occur between two end-points. We hold this in a
// simple Range type.
//
// While the parser only supports up to 2 bridges, the solver itself
// does not have this constraint.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct Range {
    min: usize,
    max: usize,
}

// An island has 4 neighbour directions, for which we track the number
// of bridges, and a valence - the total number of bridges it has. The
// bridge array goes N S W E, so that "idx ^ 1" gives the index of the
// bridge going the other way.
#[derive(Debug, Eq, PartialEq)]
struct Island {
   bridges: [Range;4],
   valence: usize,
}

// A cell in the map may be an island, a bridge (horizontal or
// vertical), or empty. This enum captures that.
#[derive(Debug, Eq, PartialEq)]
enum Cell {
    Empty,
    Island(Island),
    HBridge(usize),
    VBridge(usize),
}

// We simply represent the map as a vector of vectors (outer index is
// N-S, inner index is W-E, [0][0] represents the NW corner). We leave a
// lot of room for optimisation with clever data structures, but it's
// simple. I like simple.
struct Map(Vec<Vec<Cell>>);

////////////////////////////////////////////////////////////////////////
// Main entry point
//

fn main() {
    // TODO!
    println!("Hello, world!");
}
