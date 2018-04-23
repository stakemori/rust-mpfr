#![crate_name = "rust_mpfr"]
#![allow(non_camel_case_types)]

extern crate gmp;
extern crate libc;
extern crate serde;

#[macro_export]
macro_rules! mpfr {
    ($lit:expr) => {
        $crate::mpfr::Mpfr::new_from_str(stringify!($lit), 10)
            .expect("Invalid floating point literal")
    };
}

#[macro_use]
mod macros;
pub mod mpfr;
