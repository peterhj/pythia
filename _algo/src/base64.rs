// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub use self::CharacterSet::*;

/// Configuration for RFC 4648 standard base64 encoding
pub static STANDARD: Base64Format =
    Base64Format{char_set: Standard, newline: Newline::CRLF, pad: true, line_length: None};

/// Configuration for RFC 4648 base64url encoding
pub static URL_SAFE: Base64Format =
    Base64Format{char_set: UrlSafe, newline: Newline::CRLF, pad: false, line_length: None};

/// Configuration for RFC 2045 MIME base64 encoding
pub static MIME: Base64Format =
    Base64Format{char_set: Standard, newline: Newline::CRLF, pad: true, line_length: Some(76)};

pub static STANDARD_CHARS: &'static [u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ\
                                             abcdefghijklmnopqrstuvwxyz\
                                             0123456789+/";

pub static URL_SAFE_CHARS: &'static [u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ\
                                             abcdefghijklmnopqrstuvwxyz\
                                             0123456789-_";

/// Available encoding character sets
#[derive(Clone, Copy, Debug)]
pub enum CharacterSet {
    /// The standard character set (uses `+` and `/`)
    Standard,
    /// The URL safe character set (uses `-` and `_`)
    UrlSafe
}

/// Available newline types
#[derive(Clone, Copy, Debug)]
pub enum Newline {
    /// A linefeed (i.e. Unix-style newline)
    LF,
    /// A carriage return and a linefeed (i.e. Windows-style newline)
    CRLF
}

/// Contains configuration parameters for `to_base64`.
#[derive(Clone, Copy, Debug)]
pub struct Base64Format {
    /// Character set to use
    pub char_set: CharacterSet,
    /// Newline to use
    pub newline: Newline,
    /// True to pad output with `=` characters
    pub pad: bool,
    /// `Some(len)` to wrap lines at `len`, `None` to disable line wrapping
    pub line_length: Option<usize>
}

impl Default for Base64Format {
    fn default() -> Base64Format {
        STANDARD
    }
}

impl Base64Format {
    /// Convert a value to base64 encoding with the given configuration.
    pub fn to_string(&self, value: &[u8]) -> String {
        encode_to_base64(value, self)
    }
}

pub fn encode_to_base64(value: &[u8], config: &Base64Format) -> String {
    let bytes = match config.char_set {
        Standard => STANDARD_CHARS,
        UrlSafe => URL_SAFE_CHARS
    };

    let len = value.len();
    let newline = match config.newline {
        Newline::LF => "\n",
        Newline::CRLF => "\r\n",
    };

    // Deal with padding bytes
    let mod_len = len % 3;

    // Preallocate memory.
    let mut prealloc_len = if config.pad {
        ((len + 2) / 3) * 4
    } else {
        match mod_len {
            0 => (len / 3) * 4,
            1 => (len / 3) * 4 + 2,
            2 => (len / 3) * 4 + 3,
            _ => panic!("Algebra is broken, please alert the math police")
        }
    };
    if let Some(line_length) = config.line_length {
        let num_line_breaks = match prealloc_len {
            0 => 0,
            n => (n - 1) / line_length
        };
        prealloc_len += num_line_breaks * newline.bytes().count();
    }

    // SAFETY: The use of `set_len` below is safe, as `u8` is `Copy` and
    // the capacity was already initialized to `prealloc_len`.
    let mut out_bytes: Vec<u8> = Vec::with_capacity(prealloc_len);
    unsafe { out_bytes.set_len(prealloc_len); }

    // Use iterators to reduce branching
    {
        let mut cur_length = 0;

        let mut s_in = value[ .. len - mod_len].iter().map(|&x| x as u32);
        let mut s_out = out_bytes.iter_mut();

        // Convenient shorthand
        let enc = |val| bytes[val as usize];
        let mut write = |val| {
            // Line break if needed
            if let Some(line_length) = config.line_length {
                if cur_length >= line_length {
                    for b in newline.bytes() {
                        *s_out.next().unwrap() = b;
                    }
                    cur_length = 0;
                }
            }
            *s_out.next().unwrap() = val;
            cur_length += 1;
        };

        // Iterate though blocks of 4
        loop {
            let first = match s_in.next() {
                None => break,
                Some(first) => first
            };
            let second = s_in.next().unwrap();
            let third = s_in.next().unwrap();
            let n = (first << 16) | (second << 8) | third;
            // This 24-bit number gets separated into four 6-bit numbers.
            write(enc((n >> 18) & 63));
            write(enc((n >> 12) & 63));
            write(enc((n >>  6) & 63));
            write(enc((n      ) & 63));
        }

        // Heh, would be cool if we knew this was exhaustive
        // (the dream of bounded integer types)
        match mod_len {
            0 => {}
            1 => {
                let first = value[len - 1] as u32;
                let n = first << 16;
                write(enc((n >> 18) & 63));
                write(enc((n >> 12) & 63));
                if config.pad {
                    write(b'=');
                    write(b'=');
                }
            }
            2 => {
                let first = value[len - 2] as u32;
                let second = value[len - 1] as u32;
                let n = (first << 16) | (second << 8);
                write(enc((n >> 18) & 63));
                write(enc((n >> 12) & 63));
                write(enc((n >>  6) & 63));
                if config.pad {
                    write(b'=');
                }
            }
            _ => panic!("Algebra is broken, please alert the math police")
        }

        assert!(s_out.next().is_none());
    }

    // SAFETY: This is safe because `out_bytes` was constructed out of
    // only ASCII bytes, and we verified that it was fully initialized
    // via `s_out.next().is_none()`.
    unsafe { String::from_utf8_unchecked(out_bytes) }
}
