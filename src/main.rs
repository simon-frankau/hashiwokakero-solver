//
// Hashiwokakero solver
//
// Copyright 2021 Simon Frankau
//

use std::fs::File;
use std::io::{BufRead, BufReader, Read, stdin, stdout, Write};

use anyhow::{bail, ensure, Result};
use clap::Parser;

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
   bridges: [Range;4],
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
#[derive(Debug)]
struct Map(Vec<Vec<Cell>>);

// The (x, y) steps to move N S W E respectively.
const DIRECTION_STEPS: &[(isize, isize); 4] = &[(0, -1), (0, 1), (-1, 0), (1, 0)];

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
             bridges: [Range { min: 0, max: max_bridges }; 4],
             valence,
         }
     }

     fn bridge(&self, dir: Direction) -> Range {
         self.bridges[dir as usize]
     }

     fn bridge_mut(&mut self, dir: Direction) -> &mut Range {
         &mut self.bridges[dir as usize]
     }

}

////////////////////////////////////////////////////////////////////////
// Parser and printer
//

// Only constrained at the parsing/printing stage. The algorithm is general.
const MAX_BRIDGES: usize = 2;
const MAX_VALENCE: usize = MAX_BRIDGES * 4;

impl Cell {
    fn from_char(c: char) -> Result<Cell> {
        let max_valence = std::char::from_digit(MAX_VALENCE as u32, 10).unwrap();
        match c {
            '.' => Ok(Cell::Empty),
            '-' => Ok(Cell::HBridge(1)),
            '=' => Ok(Cell::HBridge(2)),
            '|' => Ok(Cell::VBridge(1)),
            'H' => Ok(Cell::VBridge(2)),
            i if '1' <= i && i <= max_valence =>
                Ok(Cell::Island(Island::new(MAX_BRIDGES, i.to_string().parse().unwrap()))),
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
            Cell::Island(isle) if isle.valence <= 35 =>
                std::char::from_digit(isle.valence as u32, 36).unwrap(),
            _ => '?',
       }
    }
}

fn read_map<'a, Iter: std::iter::Iterator<Item = &'a str>>(lines: Iter) -> Result<Map> {
    let map_lines = lines
        // Trim comments and whitespace
        .map(|s| s.find('#').map_or(s, |idx| &s[..idx]).trim())
        // Filter emptys lines
        .filter(|s| !s.is_empty())
        // Convert a single line
        .map(|s| s.chars().map(Cell::from_char).collect::<Result<Vec<_>>>())
        .collect::<Result<Vec<_>>>()?;

    ensure!(!map_lines.is_empty(), "Non-empty input line expected");
    let width = map_lines[0].len();
    ensure!(map_lines.iter().all(|row| row.len() == width),
        "Not all input lines were of the same length. Rectangular input expected.");

    Ok(Map(map_lines))
}

// Print results in the same format as input (less commments, etc.)
//
// NB: Lossy - just prints enough to display results - internal
// constraint information is not displayed.
fn display_map(map: &Map) -> String {
   map.0.iter().map(|row| {
       row.iter().map(|c| c.to_char()).collect::<String>()
   }).collect::<Vec<_>>().join("\n")
}

////////////////////////////////////////////////////////////////////////
// Operations on the map
//

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
                Cell::HBridge(_) => if dir.is_vertical() {
                    return None;
                },
                Cell::VBridge(_) => if dir.is_horizontal() {
                    return None;
                },
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
            // Otherwise, paint and continue.
            *cell = brush.clone();
        }
    }

    fn get_island(&mut self, x: usize, y: usize) -> &mut Island {
        if let Cell::Island(ref mut isle) = self.0[y][x] {
            isle
        } else {
            panic!("Expected island at ({}, {})", x, y);
        }
    }
}

////////////////////////////////////////////////////////////////////////
// Constraining steps
//

// Propagate constraints across bridges end-points: The min/max at one
// end of a possible bridge is constrained by the min/max at the other,
// and if there's no other end, the min/max is 0.
fn propagate_constraints(m: &mut Map, x: usize, y: usize) {
    for dir in [Direction::North, Direction::South, Direction::West, Direction::East].iter() {
        let far_end = m.find_neighbour(x, y, *dir);
        let far_range = match far_end {
            Some((far_x, far_y)) => m.get_island(far_x, far_y).bridge(dir.flip()),
            None => Range { min: 0, max: 0 },
        };

        let mut near_range = m.get_island(x, y).bridge_mut(*dir);

        // Min bridges only ratchets up with constraints, max bridges
        // ratchets down.
        near_range.min = near_range.min.max(far_range.min);
        near_range.max = near_range.max.min(far_range.max);
    }
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

    Ok(BufReader::new(file).lines().collect::<Result<Vec<_>, _>>()?)
}

fn write_output(opts: &Opts, s: &str) -> Result<()> {
   let mut file: Box<dyn Write> = match &opts.output_file {
        Some(name) => Box::new(File::create(name)?),
        None => Box::new(stdout()),
    };

    Ok(file.write_all(&s.as_bytes())?)
}

fn main() -> Result<()>{
    let opts: Opts = Opts::parse();

    ensure!(1 <= opts.max_bridges && opts.max_bridges <= 2,
        "--max_bridges must be 1 or 2");

    let input_map = read_map(read_input(&opts)?.iter().map(String::as_str))?;

    let output_string = display_map(&input_map);

    write_output(&opts, &output_string)?;

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
        assert!(read_map(input.iter().cloned()).is_err());
    }

    #[test]
    fn test_only_commments_fails() {
        let input: &[&str] = &["  # Test", "# Also test", "    ", ""];
        assert!(read_map(input.iter().cloned()).is_err());
    }

    #[test]
    fn test_unequal_lines_fails() {
        let input: &[&str] = &[".", ".."];
        assert!(read_map(input.iter().cloned()).is_err());
    }

    #[test]
    fn test_unexpected_chars_fails() {
        let input: &[&str] = &[".9."];
        assert!(read_map(input.iter().cloned()).is_err());
    }

    #[test]
    fn test_single_cell_parse() {
        for (row, expected) in [
            (".", Cell::Empty),
            ("-", Cell::HBridge(1)),
            ("=", Cell::HBridge(2)),
            ("|", Cell::VBridge(1)),
            ("H", Cell::VBridge(2)),
            ("5", Cell::Island(Island::new(MAX_BRIDGES, 5))),
        ].iter() {
            let input: &[&str] = &[row];
            let output = read_map(input.iter().cloned()).unwrap().0;
            assert_eq!(output.len(), 1);
            assert_eq!(output[0].len(), 1);
            assert_eq!(output[0][0], *expected);
        }
    }

    #[test]
    fn test_small_parse() {
        let input = "-=7\n#TEST\n|H.\n";
        let output = read_map(input.lines()).unwrap().0;
        assert_eq!(output.len(), 2);
        assert_eq!(output[0].len(), 3);
        assert_eq!(output[0][0], Cell::HBridge(1));
        assert_eq!(output[0][1], Cell::HBridge(2));
        assert_eq!(output[0][2], Cell::Island(Island::new(MAX_BRIDGES, 7)));
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
            (Cell::Island(Island::new(MAX_BRIDGES, 1)), "1"),
            // Check going over into hex
            (Cell::Island(Island::new(42, 15)), "f"),
            // Various cases of "too much"
            (Cell::Island(Island::new(42, 150)), "?"),
            (Cell::HBridge(10), "?"),
            (Cell::VBridge(10), "?"),
        ].iter() {
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
                Cell::Island(Island::new(MAX_BRIDGES, 7)),
            ],
            vec![
                Cell::VBridge(1),
                Cell::VBridge(2),
                Cell::Empty,
            ],
        ]);
        let output = display_map(&input);
        assert_eq!(output, "-=7\n|H.");
    }

    fn spaceless(s: &str) -> String {
        s.to_string().chars().filter(|c| *c != ' ').collect::<String>()
    }

    #[test]
    fn test_parse_print_round_trip() {
        // remove_matches is currently only in nightly.
        let input = spaceless("......
                               .1--2.
                               .|..H.
                               .3==4.
                               ......");
        let output = display_map(&read_map(input.lines()).unwrap());
        assert_eq!(input, output);
    }

    #[test]
    fn test_find_neighbour() {
        let input = "......
                     .1.3.5
                     ......
                     ......
                     ...2..";
        let m = read_map(input.lines()).unwrap();

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
        let mut m = read_map(input.lines()).unwrap();
        m.paint_bridge(1, 1, Direction::East, 1);
    }

    #[test]
    #[should_panic]
    fn test_paint_removal_fails() {
        let input = ".....
                     .1=2.
                     .....";
        let mut m = read_map(input.lines()).unwrap();
        m.paint_bridge(1, 1, Direction::East, 1);
    }


    #[test]
    #[should_panic]
    fn test_paint_crossing_fails() {
        let input = "...3...
                     .1.|.2.
                     ...4...";
        let mut m = read_map(input.lines()).unwrap();
        m.paint_bridge(1, 1, Direction::East, 1);
    }

    #[test]
    fn test_paint_horizontal() {
        let input = "...3...
                     .1...2.
                     ...4...";
        let mut m = read_map(input.lines()).unwrap();
        m.paint_bridge(1, 1, Direction::East, 1);

        let expected = spaceless("...3...
                                  .1---2.
                                  ...4...");
        assert_eq!(display_map(&m), expected);
    }

    #[test]
    fn test_paint_vertical() {
        let input = "...3...
                     .1...2.
                     ...4...";
        let mut m = read_map(input.lines()).unwrap();
        m.paint_bridge(3, 0, Direction::South, 1);

        let expected = spaceless("...3...
                                  .1.|.2.
                                  ...4...");
        assert_eq!(display_map(&m), expected);
    }

    #[test]
    fn test_overpaint() {
        let input = "...3...
                     .1---2.
                     ...4...";
        let mut m = read_map(input.lines()).unwrap();
        m.paint_bridge(1, 1, Direction::East, 2);

        let expected = spaceless("...3...
                                  .1===2.
                                  ...4...");
        assert_eq!(display_map(&m), expected);
    }

    #[test]
    fn test_zero_paint_noop() {
        let input = "...3...
                     .1...2.
                     ...4...";
        let mut m = read_map(input.lines()).unwrap();
        m.paint_bridge(5, 1, Direction::West, 0);

        let expected = spaceless(input);
        assert_eq!(display_map(&m), expected);
    }

    #[test]
    fn test_same_paint_noop() {
        let input = "...3...
                     .1===2.
                     ...4...";
        let mut m = read_map(input.lines()).unwrap();
        m.paint_bridge(5, 1, Direction::West, 2);

        let expected = spaceless(input);
        assert_eq!(display_map(&m), expected);
    }

    #[test]
    fn test_propagate_constraints() {
        let input = ".....
                     .3.1.
                     .....
                     .....
                     .3.1.";
        let mut m = read_map(input.lines()).unwrap();

        // Force the max/min values on a couple of island directions..
        *m.get_island(1, 4).bridge_mut(Direction::North) = Range { max: 2, min: 2 };
        *m.get_island(3, 1).bridge_mut(Direction::West) = Range { min: 0, max: 1 };

        // Propagate these values...
        propagate_constraints(&mut m, 1, 1);

        // And check they propagated correctly.
        let isle = m.get_island(1, 1);
        assert_eq!(isle.bridge(Direction::North), Range { min: 0, max: 0 });
        assert_eq!(isle.bridge(Direction::South), Range { min: 2, max: 2 });
        assert_eq!(isle.bridge(Direction::West), Range { min: 0, max: 0 });
        assert_eq!(isle.bridge(Direction::East), Range { min: 0, max: 1 });
    }
}
