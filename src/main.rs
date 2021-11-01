//
// Hashiwokakero solver
//
// Copyright 2021 Simon Frankau
//

use anyhow::{bail, ensure, Result};

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
struct Map(Vec<Vec<Cell>>);

impl Island {
     fn new(max_bridges: usize, valence: usize) -> Island {
         Island {
             bridges: [Range { min: 0, max: max_bridges }; 4],
             valence,
         }
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
// Main entry point
//

fn main() -> Result<()>{
    // TODO!
    println!("Hello, world!");
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

    #[test]
    fn test_parse_print_round_trip() {
        // remove_matches is currently only in nightly.
        let input = "......
                     .1--2.
                     .|..H.
                     .3==4.
                     ......"
            .to_string()
            .chars()
            .filter(|c| *c != ' ')
            .collect::<String>();
        let output = display_map(&read_map(input.lines()).unwrap());
        assert_eq!(input, output);
    }
}
