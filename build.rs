extern crate cc;

fn main() {
    println!("cargo:rustc-link-lib=gmp");

    cc::Build::new()
        .file("src/mpfr_wrapper.c")
        .flag("-lmpfr -lgmp")
        .compile("libmpfr_wrapper.a");
}
