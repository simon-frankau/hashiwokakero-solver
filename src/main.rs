//
// Hashiwokakero solver
//
// Copyright 2021 Simon Frankau
//

use std::collections::HashSet;
use std::fs::File;
use std::io::{stdin, stdout, BufRead, BufReader, Read, Write};

use anyhow::{bail, ensure, Result};
use clap::Parser;
use thiserror::Error;

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

// There are 4 directions from an island.
#[derive(Copy, Clone, Debug, Eq, PartialEq, PartialOrd, Ord)]
enum Direction {
    North = 0,
    South = 1,
    West = 2,
    East = 3,
}

// An island has 4 neighbour directions, for which we track the number
// of bridges, and a valence - the total number of bridges it has.
#[derive(Clone, Debug, Eq, PartialEq)]
struct Island {
    bridges: [Range; 4],
    valence: usize,
}

// A cell in the map may be an island, a bridge (horizontal or
// vertical), or empty. This enum captures that.
#[derive(Clone, Debug, Eq, PartialEq)]
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
#[derive(Clone, Debug, Eq, PartialEq)]
struct Map(Vec<Vec<Cell>>);

const ALL_DIRS: &[Direction] = &[
    Direction::North,
    Direction::South,
    Direction::West,
    Direction::East,
];

// The (x, y) steps to move N S W E respectively.
const DIRECTION_STEPS: &[(isize, isize); 4] = &[(0, -1), (0, 1), (-1, 0), (1, 0)];

#[derive(Debug, Eq, Error, PartialEq)]
enum ConstraintSolverFailure {
    // There's some kind of contradiction, meaning no solution is possible.
    #[error("No solution is possible for this configuration")]
    NoSolutions,
    // We can't make further progress. Either this is a tricky case the
    // constraint solved can't handle, or there are multiple solutions,
    // meaning we can't find a unique path forwards.
    #[error("No unique solution can be found by constraint-solving")]
    Stuck,
}

impl Direction {
    fn step(&self) -> (isize, isize) {
        DIRECTION_STEPS[*self as usize]
    }

    fn is_horizontal(&self) -> bool {
        *self >= Direction::West
    }

    fn is_vertical(&self) -> bool {
        *self <= Direction::South
    }

    fn flip(&self) -> Direction {
        // self ^ 1 also flips direction, but Rust doesn't provide a simple
        // way to go from integer to enum entry.
        match *self {
            Direction::North => Direction::South,
            Direction::South => Direction::North,
            Direction::West => Direction::East,
            Direction::East => Direction::West,
        }
    }
}

impl Island {
    fn new(max_bridges: usize, valence: usize) -> Island {
        Island {
            bridges: [Range {
                min: 0,
                max: max_bridges,
            }; 4],
            valence,
        }
    }

    fn bridge(&self, dir: Direction) -> Range {
        self.bridges[dir as usize]
    }

    fn bridge_mut(&mut self, dir: Direction) -> &mut Range {
        &mut self.bridges[dir as usize]
    }

    fn is_complete(&self) -> bool {
        self.bridges.iter().all(|r| r.min == r.max)
    }
}

////////////////////////////////////////////////////////////////////////
// Parser and printer
//

// Only constrained at the parsing/printing stage. The algorithm is general.
#[cfg(test)]
const DEFAULT_MAX_BRIDGES: usize = 2;

impl Cell {
    fn from_char(max_bridges: usize, c: char) -> Result<Cell> {
        let max_valence = std::char::from_digit((max_bridges * 4) as u32, 10).unwrap();
        match c {
            '.' => Ok(Cell::Empty),
            '-' => Ok(Cell::HBridge(1)),
            '=' => Ok(Cell::HBridge(2)),
            '|' => Ok(Cell::VBridge(1)),
            'H' => Ok(Cell::VBridge(2)),
            i if '1' <= i && i <= max_valence => Ok(Cell::Island(Island::new(
                max_bridges,
                i.to_string().parse().unwrap(),
            ))),
            _ => bail!("Unexpected character in input: '{}'", c),
        }
    }

    // NB: Undisplayable cells (valence > 35, bridge count > 2) are silently converted to '?'.
    fn to_char(&self) -> char {
        match self {
            Cell::Empty => '.',
            Cell::HBridge(1) => '-',
            Cell::HBridge(2) => '=',
            Cell::VBridge(1) => '|',
            Cell::VBridge(2) => 'H',
            Cell::Island(isle) if isle.valence <= 35 => {
                std::char::from_digit(isle.valence as u32, 36).unwrap()
            }
            _ => '?',
        }
    }
}

fn read_map<'a, Iter: std::iter::Iterator<Item = &'a str>>(
    max_bridges: usize,
    lines: Iter,
) -> Result<Map> {
    let map_lines = lines
        // Trim comments and whitespace
        .map(|s| s.find('#').map_or(s, |idx| &s[..idx]).trim())
        // Filter emptys lines
        .filter(|s| !s.is_empty())
        // Convert a single line
        .map(|s| {
            s.chars()
                .map(|c| Cell::from_char(max_bridges, c))
                .collect::<Result<Vec<_>>>()
        })
        .collect::<Result<Vec<_>>>()?;

    ensure!(!map_lines.is_empty(), "Non-empty input line expected");
    let width = map_lines[0].len();
    ensure!(
        map_lines.iter().all(|row| row.len() == width),
        "Not all input lines were of the same length. Rectangular input expected."
    );

    Ok(Map(map_lines))
}

// Print results in the same format as input (less commments, etc.)
//
// NB: Lossy - just prints enough to display results - internal
// constraint information is not displayed.
fn display_map(map: &Map) -> String {
    map.0
        .iter()
        .map(|row| row.iter().map(|c| c.to_char()).collect::<String>())
        .collect::<Vec<_>>()
        .join("\n")
}

////////////////////////////////////////////////////////////////////////
// Operations on the map
//

struct IslandIterator<'a> {
    map: &'a Map,
    width: usize,
    height: usize,
    x: usize,
    y: usize,
}

impl Map {
    fn find_neighbour(&self, x: usize, y: usize, dir: Direction) -> Option<(usize, usize)> {
        let cells = &self.0;
        let (mut idx_x, mut idx_y) = (x as isize, y as isize);
        let (len_x, len_y) = (cells[0].len() as isize, cells.len() as isize);
        let (step_x, step_y) = dir.step();
        loop {
            idx_x += step_x;
            idx_y += step_y;
            if idx_x < 0 || idx_x >= len_x || idx_y < 0 || idx_y >= len_y {
                return None;
            }
            match cells[idx_y as usize][idx_x as usize] {
                Cell::HBridge(_) => {
                    if dir.is_vertical() {
                        return None;
                    }
                }
                Cell::VBridge(_) => {
                    if dir.is_horizontal() {
                        return None;
                    }
                }
                Cell::Island(_) => return Some((idx_x as usize, idx_y as usize)),
                Cell::Empty => (),
            }
        }
    }

    fn paint_bridge(&mut self, x: usize, y: usize, dir: Direction, n_bridges: usize) {
        let brush = if dir.is_horizontal() {
            Cell::HBridge(n_bridges)
        } else {
            Cell::VBridge(n_bridges)
        };

        let cells = &mut self.0;
        let (mut idx_x, mut idx_y) = (x as isize, y as isize);
        let (len_x, len_y) = (cells[0].len() as isize, cells.len() as isize);
        let (step_x, step_y) = dir.step();

        // This could be a lot simpler, but we'll be paranoid about
        // painting over the wrong thing.
        loop {
            idx_x += step_x;
            idx_y += step_y;
            assert!(0 <= idx_x && idx_x < len_x);
            assert!(0 <= idx_y && idx_y < len_y);

            let cell = &mut cells[idx_y as usize][idx_x as usize];
            // How many bridges there already?
            let n_existing: usize = match cell {
                Cell::Empty => 0,
                Cell::HBridge(n) => {
                    assert!(dir.is_horizontal());
                    *n
                }
                Cell::VBridge(n) => {
                    assert!(dir.is_vertical());
                    *n
                }
                // If we hit an island, we're done.
                Cell::Island(_) => return,
            };

            // We only add bridges.
            assert!(n_existing <= n_bridges);
            // Right number already present - we're done.
            if n_existing == n_bridges {
                return;
            }
            // Otherwise, paint and continue.<
            *cell = brush.clone();
        }
    }

    fn get_island(&self, x: usize, y: usize) -> &Island {
        if let Cell::Island(ref isle) = self.0[y][x] {
            isle
        } else {
            panic!("Expected island at ({}, {})", x, y);
        }
    }

    fn get_island_mut(&mut self, x: usize, y: usize) -> &mut Island {
        if let Cell::Island(ref mut isle) = self.0[y][x] {
            isle
        } else {
            panic!("Expected island at ({}, {})", x, y);
        }
    }

    // Helper function to is_solved.
    // Returns number of complete components, and incomplete subgraphs.
    fn count_components(&self) -> (usize, usize) {
        let mut seen = HashSet::new();
        let mut complete = 0;
        let mut incomplete = 0;

        // 'aux' depth-first searches through complete nodes, trying
        // to find maximal subgraphs of complete nodes. If the nodes
        // in such a subgraph have no incomplete neighbours, we have a
        // complete component and the complete counter is increased by
        // count_components.
        //
        // Otherwise this subgraph is counted as one incomplete subgraph.
        fn aux(
            map: &Map,
            x: usize,
            y: usize,
            seen: &mut HashSet<(usize, usize)>,
            is_complete: &mut bool,
        ) {
            if !map.get_island(x, y).is_complete() {
                *is_complete = false;
                return;
            }

            // This island is complete. Nothing to do if we've processed
            // it already.
            if seen.contains(&(x, y)) {
                return;
            }

            // Otherwise, mark as seen and recurse.
            seen.insert((x, y));
            for dir in ALL_DIRS.iter().cloned() {
                if map.get_island(x, y).bridge(dir).min > 0 {
                    if let Some((nx, ny)) = map.find_neighbour(x, y, dir) {
                        aux(map, nx, ny, seen, is_complete);
                    }
                }
            }
        }

        for (x, y) in self.island_iter() {
            if !seen.contains(&(x, y)) {
                // We have found the potential root of a new component.
                let mut is_complete = true;
                aux(self, x, y, &mut seen, &mut is_complete);
                seen.insert((x, y));
                if is_complete {
                    complete += 1;
                } else {
                    incomplete += 1;
                }
            }
        }

        (complete, incomplete)
    }

    // We have a solution if there's a single complete component.
    // Returns insoluble if there's a complete component and anything
    // left over, since we can't extend the solution to a single
    // complete component.
    fn is_solved(&self) -> Result<bool, ConstraintSolverFailure> {
        let (complete, incomplete) = self.count_components();
        let total = complete + incomplete;

        if complete > 0 {
            // If we have a complete component, it must be unique in order
            // to have a solution.
            if total == 1 {
                Ok(true)
            } else {
                Err(ConstraintSolverFailure::NoSolutions)
            }
        } else {
            // No complete component? No solution.
            Ok(false)
        }
    }

    fn island_iter(&self) -> IslandIterator {
        IslandIterator::new(self)
    }
}

impl<'a> IslandIterator<'a> {
    fn new(map: &'a Map) -> IslandIterator<'a> {
        let cells = &map.0;
        let (width, height) = (cells[0].len(), cells.len());
        IslandIterator {
            map,
            width,
            height,
            x: 0,
            y: 0,
        }
    }

    fn step(&mut self) {
        self.x += 1;
        if self.x >= self.width {
            self.x = 0;
            self.y += 1;
        }
    }
}

impl<'a> Iterator for IslandIterator<'a> {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.y >= self.height {
                return None;
            }
            let (x, y) = (self.x, self.y);
            self.step();
            if let Cell::Island(_) = self.map.0[y][x] {
                return Some((x, y));
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////
// Constraining steps
//

// Propagate constraints across bridges end-points: The min/max at one
// end of a possible bridge is constrained by the min/max at the other,
// and if there's no other end, the min/max is 0.
fn propagate_constraints(m: &mut Map, x: usize, y: usize) -> Result<(), ConstraintSolverFailure> {
    for dir in ALL_DIRS.iter() {
        let far_end = m.find_neighbour(x, y, *dir);
        let far_range = match far_end {
            Some((far_x, far_y)) => m.get_island(far_x, far_y).bridge(dir.flip()),
            None => Range { min: 0, max: 0 },
        };

        let mut near_range = m.get_island_mut(x, y).bridge_mut(*dir);

        // Min bridges only ratchets up with constraints, max bridges
        // ratchets down.
        near_range.min = near_range.min.max(far_range.min);
        near_range.max = near_range.max.min(far_range.max);

        // If the values cross over, something's gone wrong!
        if near_range.min > near_range.max {
            return Err(ConstraintSolverFailure::NoSolutions);
        }
    }

    Ok(())
}

// Generate all the ways to distribute the valence across the set of
// bridge endpoints - returns a list of allocations such that the
// value for each endpoint is between min and max, and they sum to the
// valence.
fn generate_distributions(isle: &Island) -> Vec<[usize; 4]> {
    // There's no clean way to make recursive closures, so we live with
    // a nested function and passing all variables explicitly.
    fn aux(
        isle: &Island,
        acc: &mut Vec<[usize; 4]>,
        mut curr: [usize; 4],
        idx: usize,
        remaining: usize,
    ) {
        if idx == 4 {
            if remaining == 0 {
                acc.push(curr);
            }
            return;
        }

        let r = isle.bridges[idx];
        if r.min > remaining {
            return;
        }
        for n in (r.min)..=(r.max.min(remaining)) {
            curr[idx] = n;
            aux(isle, acc, curr, idx + 1, remaining - n);
        }
    }

    let mut v: Vec<[usize; 4]> = Vec::new();
    aux(isle, &mut v, [0; 4], 0, isle.valence);
    v
}

// Given the valence, update the min/max ranges. This implements the
// pigeonhole principle constraints - if we have 2 neighbours, and
// a maximum of 2 bridges per neighbour, and a valence of 3, we
// *must* have at least one bridge per neighbour.
fn apply_valence_constraints(isle: &mut Island) -> Result<(), ConstraintSolverFailure> {
    let distributions = generate_distributions(isle);

    if distributions.is_empty() {
        return Err(ConstraintSolverFailure::NoSolutions);
    }

    // The distributions are already guaranteed to be within the
    // min/max bounds, so the updated bounds will just be the min/max
    // of the generated possibilities
    for idx in 0..isle.bridges.len() {
        let min = distributions.iter().map(|v| v[idx]).min().unwrap();
        let max = distributions.iter().map(|v| v[idx]).max().unwrap();
        isle.bridges[idx] = Range { min, max }
    }

    Ok(())
}

////////////////////////////////////////////////////////////////////////
// Solving algorithm
//

// Perform a constraints update on a single island. Failure to
// constrain further is not an error, we just return true if constraints
// were tightened further, and false otherwise.
fn constrain_island(m: &mut Map, x: usize, y: usize) -> Result<bool, ConstraintSolverFailure> {
    let before = m.get_island(x, y).clone();

    propagate_constraints(m, x, y)?;
    let island = m.get_island_mut(x, y);
    apply_valence_constraints(island)?;

    let changed = *island != before;

    if changed {
        // Need to copy as paint_bridge uses a mutable borrow of m.
        let after = island.clone();

        for dir in ALL_DIRS.iter() {
            let range = after.bridge(*dir);
            if range.min == range.max && range.min != 0 {
                // There is some bridge. Ensure it's up-to-date.
                m.paint_bridge(x, y, *dir, range.min);
            }
        }
    }

    Ok(changed)
}

// Perform a single constraint-solving pass over the entire map.
fn constrain_pass(m: &mut Map) -> Result<(), ConstraintSolverFailure> {
    let mut progress = false;

    // Make a copy as constrain_island updates the map.
    let islands = m.island_iter().collect::<Vec<_>>();
    for (x, y) in islands.iter() {
        progress |= constrain_island(m, *x, *y)?;
    }

    if !progress {
        Err(ConstraintSolverFailure::Stuck)
    } else {
        Ok(())
    }
}

// Constrain until a solution or failure or stuckness is reached.
fn constrain(m: &mut Map) -> Result<(), ConstraintSolverFailure> {
    while !m.is_solved()? {
        constrain_pass(m)?;
    }
    Ok(())
}

// If we do something clever around choosing what to case split on,
// this is where we'd do it. As such, it's a bit of a placeholder right
// now, and just finds the very first possibility.
//
// Returns (x, y, dir) to do the case split on.
fn choose_case_split(m: &Map) -> (usize, usize, Direction) {
    for (x, y) in m.island_iter() {
        let isle = m.get_island(x, y);
        for dir in ALL_DIRS.iter() {
            let range = isle.bridge(*dir);
            if range.min != range.max {
                // Multiple possibilities, can perform case split here.
                return (x, y, *dir);
            }
        }
    }

    panic!("Shouldn't happen: Nothing to case split on.")
}

// Find all solutions by applying case splits as necessary. Works by
// vanilla recursion, so you need a reasonable size stack for lots of
// splits (although if you're searching a large number of splits,
// it'll probably be quite slow!).
//
// This is an auxiliary function to solve, and takes ownership of the
// provided map.
fn accumulate_solutions(mut map: Map, solutions: &mut Vec<Map>) {
    match constrain(&mut map) {
        Ok(()) => solutions.push(map),
        Err(ConstraintSolverFailure::NoSolutions) => (),
        Err(ConstraintSolverFailure::Stuck) => {
            let (x, y, dir) = choose_case_split(&map);
            let range = map.get_island(x, y).bridge(dir);
            for n in range.min..=range.max {
                let mut new_map = map.clone();
                let bridge = new_map.get_island_mut(x, y).bridge_mut(dir);
                *bridge = Range { min: n, max: n };
                accumulate_solutions(new_map, solutions);
            }
        }
    }
}

// Find all solutions.
fn solve(map: &Map) -> Vec<Map> {
    let mut solutions = Vec::new();
    accumulate_solutions(map.clone(), &mut solutions);
    solutions
}

////////////////////////////////////////////////////////////////////////
// Main entry point
//

#[derive(Parser)]
#[clap(version = "0.1", author = "Simon Frankau <sgf@arbitrary.name>")]
#[clap(about = "Hashiwokakero (Bridges) puzzle solver")]
struct Opts {
    /// Input file. Uses stdin if none specified.
    #[clap(long)]
    input_file: Option<String>,
    /// Output file. Uses stdin if none specified.
    #[clap(long)]
    output_file: Option<String>,
    /// Maximum number of bridges between islands.
    #[clap(long, default_value = "2")]
    max_bridges: usize,
}

fn read_input(opts: &Opts) -> Result<Vec<String>> {
    let file: Box<dyn Read> = match &opts.input_file {
        Some(name) => Box::new(File::open(name)?),
        None => Box::new(stdin()),
    };

    Ok(BufReader::new(file)
        .lines()
        .collect::<Result<Vec<_>, _>>()?)
}

fn write_output(opts: &Opts, s: &str) -> Result<()> {
    let mut file: Box<dyn Write> = match &opts.output_file {
        Some(name) => Box::new(File::create(name)?),
        None => Box::new(stdout()),
    };

    Ok(file.write_all(s.as_bytes())?)
}

fn main() -> Result<()> {
    let opts: Opts = Opts::parse();

    ensure!(
        1 <= opts.max_bridges && opts.max_bridges <= 2,
        "--max_bridges must be 1 or 2"
    );

    let input_map = read_map(
        opts.max_bridges,
        read_input(&opts)?.iter().map(String::as_str),
    )?;

    let solutions = solve(&input_map);

    if solutions.is_empty() {
        eprintln!("No solutions");
    } else {
        let output_string = solutions
            .iter()
            .map(display_map)
            .collect::<Vec<_>>()
            .join("\n\n");

        write_output(&opts, &output_string)?;
    }

    Ok(())
}

////////////////////////////////////////////////////////////////////////
// Tests
//

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_completely_empty_fails() {
        let input: &[&str] = &[];
        assert!(read_map(DEFAULT_MAX_BRIDGES, input.iter().cloned()).is_err());
    }

    #[test]
    fn test_only_commments_fails() {
        let input: &[&str] = &["  # Test", "# Also test", "    ", ""];
        assert!(read_map(DEFAULT_MAX_BRIDGES, input.iter().cloned()).is_err());
    }

    #[test]
    fn test_unequal_lines_fails() {
        let input: &[&str] = &[".", ".."];
        assert!(read_map(DEFAULT_MAX_BRIDGES, input.iter().cloned()).is_err());
    }

    #[test]
    fn test_unexpected_chars_fails() {
        let input: &[&str] = &[".9."];
        assert!(read_map(DEFAULT_MAX_BRIDGES, input.iter().cloned()).is_err());
    }

    #[test]
    fn test_single_cell_parse() {
        for (row, expected) in [
            (".", Cell::Empty),
            ("-", Cell::HBridge(1)),
            ("=", Cell::HBridge(2)),
            ("|", Cell::VBridge(1)),
            ("H", Cell::VBridge(2)),
            ("5", Cell::Island(Island::new(DEFAULT_MAX_BRIDGES, 5))),
        ]
        .iter()
        {
            let input: &[&str] = &[row];
            let output = read_map(DEFAULT_MAX_BRIDGES, input.iter().cloned())
                .unwrap()
                .0;
            assert_eq!(output.len(), 1);
            assert_eq!(output[0].len(), 1);
            assert_eq!(output[0][0], *expected);
        }
    }

    #[test]
    fn test_small_parse() {
        let input = "-=7\n#TEST\n|H.\n";
        let output = read_map(DEFAULT_MAX_BRIDGES, input.lines()).unwrap().0;
        assert_eq!(output.len(), 2);
        assert_eq!(output[0].len(), 3);
        assert_eq!(output[0][0], Cell::HBridge(1));
        assert_eq!(output[0][1], Cell::HBridge(2));
        assert_eq!(
            output[0][2],
            Cell::Island(Island::new(DEFAULT_MAX_BRIDGES, 7))
        );
        assert_eq!(output[1].len(), 3);
        assert_eq!(output[1][0], Cell::VBridge(1));
        assert_eq!(output[1][1], Cell::VBridge(2));
        assert_eq!(output[1][2], Cell::Empty);
    }

    #[test]
    fn test_single_cell_print() {
        for (cell, expected) in [
            (Cell::Empty, "."),
            (Cell::HBridge(1), "-"),
            (Cell::HBridge(2), "="),
            (Cell::VBridge(1), "|"),
            (Cell::VBridge(2), "H"),
            (Cell::Island(Island::new(DEFAULT_MAX_BRIDGES, 1)), "1"),
            // Check going over into hex
            (Cell::Island(Island::new(42, 15)), "f"),
            // Various cases of "too much"
            (Cell::Island(Island::new(42, 150)), "?"),
            (Cell::HBridge(10), "?"),
            (Cell::VBridge(10), "?"),
        ]
        .iter()
        {
            let input = Map(vec![vec![cell.clone()]]);
            let output = display_map(&input);
            assert_eq!(&output, expected);
        }
    }

    #[test]
    fn test_small_print() {
        let input = Map(vec![
            vec![
                Cell::HBridge(1),
                Cell::HBridge(2),
                Cell::Island(Island::new(DEFAULT_MAX_BRIDGES, 7)),
            ],
            vec![Cell::VBridge(1), Cell::VBridge(2), Cell::Empty],
        ]);
        let output = display_map(&input);
        assert_eq!(output, "-=7\n|H.");
    }

    fn spaceless(s: &str) -> String {
        s.to_string()
            .chars()
            .filter(|c| *c != ' ')
            .collect::<String>()
    }

    #[test]
    fn test_parse_print_round_trip() {
        // remove_matches is currently only in nightly.
        let input = spaceless(
            "......
             .1--2.
             .|..H.
             .3==4.
             ......",
        );
        let output = display_map(&read_map(DEFAULT_MAX_BRIDGES, input.lines()).unwrap());
        assert_eq!(input, output);
    }

    #[test]
    fn test_find_neighbour() {
        let input = "......
                     .1.3.5
                     ......
                     ......
                     ...2..";
        let m = read_map(DEFAULT_MAX_BRIDGES, input.lines()).unwrap();

        assert_eq!(m.find_neighbour(3, 1, Direction::North), None);
        assert_eq!(m.find_neighbour(3, 1, Direction::South), Some((3, 4)));

        assert_eq!(m.find_neighbour(3, 1, Direction::West), Some((1, 1)));
        assert_eq!(m.find_neighbour(3, 1, Direction::East), Some((5, 1)));

        assert_eq!(m.find_neighbour(5, 1, Direction::East), None);
        assert_eq!(m.find_neighbour(3, 4, Direction::South), None);
    }

    #[test]
    #[should_panic]
    fn test_paint_off_edge_fails() {
        let input = "...
                     .1.
                     ...";
        let mut m = read_map(DEFAULT_MAX_BRIDGES, input.lines()).unwrap();
        m.paint_bridge(1, 1, Direction::East, 1);
    }

    #[test]
    #[should_panic]
    fn test_paint_removal_fails() {
        let input = ".....
                     .1=2.
                     .....";
        let mut m = read_map(DEFAULT_MAX_BRIDGES, input.lines()).unwrap();
        m.paint_bridge(1, 1, Direction::East, 1);
    }

    #[test]
    #[should_panic]
    fn test_paint_crossing_fails() {
        let input = "...3...
                     .1.|.2.
                     ...4...";
        let mut m = read_map(DEFAULT_MAX_BRIDGES, input.lines()).unwrap();
        m.paint_bridge(1, 1, Direction::East, 1);
    }

    #[test]
    fn test_paint_horizontal() {
        let input = "...3...
                     .1...2.
                     ...4...";
        let mut m = read_map(DEFAULT_MAX_BRIDGES, input.lines()).unwrap();
        m.paint_bridge(1, 1, Direction::East, 1);

        let expected = spaceless(
            "...3...
             .1---2.
             ...4...",
        );
        assert_eq!(display_map(&m), expected);
    }

    #[test]
    fn test_paint_vertical() {
        let input = "...3...
                     .1...2.
                     ...4...";
        let mut m = read_map(DEFAULT_MAX_BRIDGES, input.lines()).unwrap();
        m.paint_bridge(3, 0, Direction::South, 1);

        let expected = spaceless(
            "...3...
             .1.|.2.
             ...4...",
        );
        assert_eq!(display_map(&m), expected);
    }

    #[test]
    fn test_overpaint() {
        let input = "...3...
                     .1---2.
                     ...4...";
        let mut m = read_map(DEFAULT_MAX_BRIDGES, input.lines()).unwrap();
        m.paint_bridge(1, 1, Direction::East, 2);

        let expected = spaceless(
            "...3...
             .1===2.
             ...4...",
        );
        assert_eq!(display_map(&m), expected);
    }

    #[test]
    fn test_zero_paint_noop() {
        let input = "...3...
                     .1...2.
                     ...4...";
        let mut m = read_map(DEFAULT_MAX_BRIDGES, input.lines()).unwrap();
        m.paint_bridge(5, 1, Direction::West, 0);

        let expected = spaceless(input);
        assert_eq!(display_map(&m), expected);
    }

    #[test]
    fn test_same_paint_noop() {
        let input = "...3...
                     .1===2.
                     ...4...";
        let mut m = read_map(DEFAULT_MAX_BRIDGES, input.lines()).unwrap();
        m.paint_bridge(5, 1, Direction::West, 2);

        let expected = spaceless(input);
        assert_eq!(display_map(&m), expected);
    }

    #[test]
    fn test_propagate_constraints() -> Result<()> {
        let input = ".....
                     .3.1.
                     .....
                     .....
                     .3.1.";
        let mut m = read_map(DEFAULT_MAX_BRIDGES, input.lines()).unwrap();

        // Force the max/min values on a couple of island directions..
        *m.get_island_mut(1, 4).bridge_mut(Direction::North) = Range { max: 2, min: 2 };
        *m.get_island_mut(3, 1).bridge_mut(Direction::West) = Range { min: 0, max: 1 };

        // Propagate these values...
        propagate_constraints(&mut m, 1, 1)?;

        // And check they propagated correctly.
        let isle = m.get_island(1, 1);
        assert_eq!(isle.bridge(Direction::North), Range { min: 0, max: 0 });
        assert_eq!(isle.bridge(Direction::South), Range { min: 2, max: 2 });
        assert_eq!(isle.bridge(Direction::West), Range { min: 0, max: 0 });
        assert_eq!(isle.bridge(Direction::East), Range { min: 0, max: 1 });

        Ok(())
    }

    #[test]
    fn test_propagate_constraints_failure() {
        let input = ".3.1.";
        let mut m = read_map(DEFAULT_MAX_BRIDGES, input.lines()).unwrap();

        // Force the max/min values on a couple of island directions..
        *m.get_island_mut(1, 0).bridge_mut(Direction::East) = Range { max: 3, min: 3 };
        *m.get_island_mut(3, 0).bridge_mut(Direction::West) = Range { min: 1, max: 1 };

        // Propagate these values. Should fail.
        match propagate_constraints(&mut m, 1, 0) {
            Result::Err(ConstraintSolverFailure::NoSolutions) => (),
            e => panic!("Unexpected result: {:?}", e),
        }
    }

    #[test]
    fn test_generate_distributions() {
        let isle = Island {
            bridges: [
                Range { min: 0, max: 2 },
                Range { min: 0, max: 2 },
                Range { min: 1, max: 2 },
                Range { min: 0, max: 2 },
            ],
            valence: 4,
        };

        let mut dists = generate_distributions(&isle);
        // Avoid making assumptions about the order it generates items in.
        dists.sort();

        let expected = [
            [0, 0, 2, 2],
            [0, 1, 1, 2],
            [0, 1, 2, 1],
            [0, 2, 1, 1],
            [0, 2, 2, 0],
            [1, 0, 1, 2],
            [1, 0, 2, 1],
            [1, 1, 1, 1],
            [1, 1, 2, 0],
            [1, 2, 1, 0],
            [2, 0, 1, 1],
            [2, 0, 2, 0],
            [2, 1, 1, 0],
        ];
        assert_eq!(&dists, &expected);
    }

    #[test]
    fn test_apply_valence_constraints_partial() -> Result<()> {
        let mut isle = Island {
            bridges: [
                Range { min: 0, max: 0 },
                Range { min: 0, max: 0 },
                Range { min: 0, max: 2 },
                Range { min: 0, max: 2 },
            ],
            valence: 3,
        };

        apply_valence_constraints(&mut isle)?;

        let expected = Island {
            bridges: [
                Range { min: 0, max: 0 },
                Range { min: 0, max: 0 },
                Range { min: 1, max: 2 },
                Range { min: 1, max: 2 },
            ],
            valence: 3,
        };
        assert_eq!(&isle, &expected);

        Ok(())
    }

    #[test]
    fn test_apply_valence_constraints_full() -> Result<()> {
        let mut isle = Island {
            bridges: [
                Range { min: 0, max: 0 },
                Range { min: 0, max: 0 },
                Range { min: 0, max: 1 },
                Range { min: 0, max: 2 },
            ],
            valence: 3,
        };

        apply_valence_constraints(&mut isle)?;

        let expected = Island {
            bridges: [
                Range { min: 0, max: 0 },
                Range { min: 0, max: 0 },
                Range { min: 1, max: 1 },
                Range { min: 2, max: 2 },
            ],
            valence: 3,
        };
        assert_eq!(&isle, &expected);

        Ok(())
    }

    #[test]
    fn test_apply_valence_constraints_failure() {
        let mut isle = Island {
            bridges: [
                Range { min: 0, max: 0 },
                Range { min: 0, max: 0 },
                Range { min: 2, max: 2 },
                Range { min: 2, max: 2 },
            ],
            valence: 3,
        };

        match apply_valence_constraints(&mut isle) {
            Result::Err(ConstraintSolverFailure::NoSolutions) => (),
            e => panic!("Unexpected result: {:?}", e),
        }
    }

    #[test]
    fn test_solve_simple() {
        let input = ".3..1
                     .....
                     .3.2.
                     .....
                     ...2.
                     .....
                     1..2.";
        let mut m = read_map(DEFAULT_MAX_BRIDGES, input.lines()).unwrap();

        assert!(constrain(&mut m).is_ok());

        let expected = spaceless(
            ".3--1
             .H...
             .3-2.
             ...|.
             ...2.
             ...|.
             1--2.",
        );
        assert_eq!(display_map(&m), expected);
    }

    #[test]
    fn test_solve_simple_no_solution() {
        let input = ".1.";
        let mut m = read_map(DEFAULT_MAX_BRIDGES, input.lines()).unwrap();

        assert_eq!(constrain(&mut m), Err(ConstraintSolverFailure::NoSolutions));
    }

    #[test]
    fn test_solve_simple_stuck() {
        let input = ".2.1.
                     .....
                     .2.2.
                     .....
                     .1.2.";
        let mut m = read_map(DEFAULT_MAX_BRIDGES, input.lines()).unwrap();

        assert_eq!(constrain(&mut m), Err(ConstraintSolverFailure::Stuck));
    }

    #[test]
    fn test_solve_split() {
        let input = ".1.2.1.
                     .......
                     .2.4.2.
                     .......
                     .1.2.1.";
        let m = read_map(DEFAULT_MAX_BRIDGES, input.lines()).unwrap();

        let sol1 = spaceless(
            ".1-2.1.
             ...|.|.
             .2-4-2.
             .|.|...
             .1.2-1.",
        );
        let sol2 = spaceless(
            ".1.2-1.
             .|.|...
             .2-4-2.
             ...|.|.
             .1-2.1.",
        );
        let mut expected = vec![sol1, sol2];
        expected.sort();

        let mut actual = solve(&m).iter().map(display_map).collect::<Vec<String>>();
        actual.sort();

        assert_eq!(expected, actual);
    }

    #[test]
    fn test_solve_split_no_solution() {
        let input = ".1.";
        let m = read_map(DEFAULT_MAX_BRIDGES, input.lines()).unwrap();

        assert!(solve(&m).is_empty());
    }

    #[test]
    fn test_solve_single_component() {
        // Check we only get the solution with a single component.
        let input = ".2.1.
                     .....
                     .2.2.
                     .....
                     .1.2.";
        let m = read_map(DEFAULT_MAX_BRIDGES, input.lines()).unwrap();

        let expected = spaceless(
            ".2-1.
             .|...
             .2-2.
             ...|.
             .1-2.",
        );

        let actual = solve(&m);
        assert_eq!(actual.len(), 1);
        assert_eq!(display_map(&actual[0]), expected);
    }

    #[test]
    fn test_solve_single_component_no_solutions() {
        let input = ".1.1
                     1.1.";
        let m = read_map(DEFAULT_MAX_BRIDGES, input.lines()).unwrap();
        assert!(solve(&m).is_empty());
    }
}
