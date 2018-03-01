use gmp::mpf::{mpf_ptr, mpf_srcptr, Mpf};
use gmp::mpq::{mpq_srcptr, Mpq};
use gmp::mpz::{mp_limb_t, mpz_ptr, mpz_srcptr, Mpz};
use libc::{c_char, c_double, c_int, c_long, c_ulong, size_t};
use serde::ser::{Serialize, Serializer};
use serde::{Deserialize, Deserializer};
use serde::de;

use std::ffi::CStr;
use std::cmp::{Ordering, PartialEq, PartialOrd};
use self::Ordering::*;
use std::cmp;
use std::convert::{From, Into};
use std::ffi::CString;
use std::fmt;
use std::mem::uninitialized;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use std::str;
use std::ptr;

type mpfr_prec_t = c_long;
type mpfr_sign_t = c_int;
type mpfr_exp_t = c_long;

#[repr(C)]
pub enum mpfr_rnd_t {
    MPFR_RNDN = 0,   // round to nearest, with ties to even
    MPFR_RNDZ,       // round toward zero
    MPFR_RNDU,       // round toward +Inf
    MPFR_RNDD,       // round toward -Inf
    MPFR_RNDA,       // round away from zero
    MPFR_RNDF,       // faithful rounding (not implemented yet)
    MPFR_RNDNA = -1, // round to nearest, with ties away from zero (mpfr_rouund)
}

#[repr(C)]
pub struct mpfr_struct {
    _mpfr_prec: mpfr_prec_t,
    _mpfr_sign: mpfr_sign_t,
    _mpfr_exp: mpfr_exp_t,
    _mpfr_d: *mut mp_limb_t,
}

pub type mpfr_srcptr = *const mpfr_struct;
pub type mpfr_ptr = *mut mpfr_struct;

const DEFAULT_RND: mpfr_rnd_t = mpfr_rnd_t::MPFR_RNDN;

#[link(name = "mpfr")]
extern "C" {
    // Initialization
    pub fn mpfr_init(x: mpfr_ptr);
    pub fn mpfr_init2(x: mpfr_ptr, prec: mpfr_prec_t);
    pub fn mpfr_clear(x: mpfr_ptr);
    pub fn mpfr_set_default_prec(prec: mpfr_prec_t);
    pub fn mpfr_get_default_prec() -> mpfr_prec_t;
    pub fn mpfr_set_prec(x: mpfr_ptr, prec: mpfr_prec_t);
    pub fn mpfr_get_prec(x: mpfr_srcptr) -> mpfr_prec_t;
    pub fn mpfr_get_exp(x: mpfr_srcptr) -> mpfr_exp_t;

    // Assignment
    pub fn mpfr_set(rop: mpfr_ptr, op: mpfr_srcptr, rnd: mpfr_rnd_t) -> c_int;
    pub fn mpfr_set_ui(rop: mpfr_ptr, op: c_ulong, rnd: mpfr_rnd_t) -> c_int;
    pub fn mpfr_set_si(rop: mpfr_ptr, op: c_long, rnd: mpfr_rnd_t) -> c_int;
    pub fn mpfr_set_d(rop: mpfr_ptr, op: c_double, rnd: mpfr_rnd_t) -> c_int;
    pub fn mpfr_set_z(rop: mpfr_ptr, op: mpz_srcptr, rnd: mpfr_rnd_t) -> c_int;
    pub fn mpfr_set_q(rop: mpfr_ptr, op: mpq_srcptr, rnd: mpfr_rnd_t) -> c_int;
    pub fn mpfr_set_f(rop: mpfr_ptr, op: mpf_srcptr, rnd: mpfr_rnd_t) -> c_int;
    pub fn mpfr_set_ui_2exp(rop: mpfr_ptr, op: c_ulong, e: mpfr_exp_t, rnd: mpfr_rnd_t) -> c_int;
    pub fn mpfr_set_si_2exp(rop: mpfr_ptr, op: c_long, e: mpfr_exp_t, rnd: mpfr_rnd_t) -> c_int;
    pub fn mpfr_set_z_2exp(rop: mpfr_ptr, op: mpz_srcptr, e: mpfr_exp_t, rnd: mpfr_rnd_t) -> c_int;
    pub fn mpfr_set_nan(x: mpfr_ptr);
    pub fn mpfr_set_inf(x: mpfr_ptr, sign: c_int);
    pub fn mpfr_set_zero(x: mpfr_ptr, sign: c_int);
    pub fn mpfr_set_str(rop: mpfr_ptr, s: *const c_char, base: c_int, rnd: mpfr_rnd_t) -> c_int;

    // Conversion
    pub fn mpfr_get_ui(op: mpfr_srcptr, rnd: mpfr_rnd_t) -> c_ulong;
    pub fn mpfr_get_si(op: mpfr_srcptr, rnd: mpfr_rnd_t) -> c_long;
    pub fn mpfr_get_d(op: mpfr_srcptr, rnd: mpfr_rnd_t) -> c_double;
    pub fn mpfr_get_z(rop: mpz_ptr, op: mpfr_srcptr, rnd: mpfr_rnd_t) -> c_int;
    pub fn mpfr_get_f(rop: mpf_ptr, op: mpfr_srcptr, rnd: mpfr_rnd_t) -> c_int;

    // Comparison
    pub fn mpfr_cmp(op1: mpfr_srcptr, op2: mpfr_srcptr) -> c_int;
    pub fn mpfr_cmp_ui(op1: mpfr_srcptr, op2: c_ulong) -> c_int;

    // Arithmetic
    pub fn mpfr_add(rop: mpfr_ptr, op1: mpfr_srcptr, op2: mpfr_srcptr, rnd: mpfr_rnd_t) -> c_int;
    pub fn mpfr_add_d(rop: mpfr_ptr, op1: mpfr_srcptr, op2: c_double, rnd: mpfr_rnd_t) -> c_int;
    pub fn mpfr_add_si(rop: mpfr_ptr, op1: mpfr_srcptr, op2: c_long, rnd: mpfr_rnd_t) -> c_int;
    pub fn mpfr_sub(rop: mpfr_ptr, op1: mpfr_srcptr, op2: mpfr_srcptr, rnd: mpfr_rnd_t) -> c_int;
    pub fn mpfr_sub_d(rop: mpfr_ptr, op1: mpfr_srcptr, op2: c_double, rnd: mpfr_rnd_t) -> c_int;
    pub fn mpfr_sub_si(rop: mpfr_ptr, op1: mpfr_srcptr, op2: c_long, rnd: mpfr_rnd_t) -> c_int;
    pub fn mpfr_d_sub(rop: mpfr_ptr, op1: c_double, op2: mpfr_srcptr, rnd: mpfr_rnd_t) -> c_int;
    pub fn mpfr_si_sub(rop: mpfr_ptr, op1: c_long, op2: mpfr_srcptr, rnd: mpfr_rnd_t) -> c_int;
    pub fn mpfr_mul(rop: mpfr_ptr, op1: mpfr_srcptr, op2: mpfr_srcptr, rnd: mpfr_rnd_t) -> c_int;
    pub fn mpfr_mul_d(rop: mpfr_ptr, op1: mpfr_srcptr, op2: c_double, rnd: mpfr_rnd_t) -> c_int;
    pub fn mpfr_mul_si(rop: mpfr_ptr, op1: mpfr_srcptr, op2: c_long, rnd: mpfr_rnd_t) -> c_int;
    pub fn mpfr_div(rop: mpfr_ptr, op1: mpfr_srcptr, op2: mpfr_srcptr, rnd: mpfr_rnd_t) -> c_int;
    pub fn mpfr_div_d(rop: mpfr_ptr, op1: mpfr_srcptr, op2: c_double, rnd: mpfr_rnd_t) -> c_int;
    pub fn mpfr_div_si(rop: mpfr_ptr, op1: mpfr_srcptr, op2: c_long, rnd: mpfr_rnd_t) -> c_int;
    pub fn mpfr_d_div(rop: mpfr_ptr, op1: c_double, op2: mpfr_srcptr, rnd: mpfr_rnd_t) -> c_int;
    pub fn mpfr_si_div(rop: mpfr_ptr, op1: c_long, op2: mpfr_srcptr, rnd: mpfr_rnd_t) -> c_int;
    pub fn mpfr_neg(rop: mpfr_ptr, op: mpfr_srcptr, rnd: mpfr_rnd_t) -> c_int;

    // Rounding
    pub fn mpfr_floor(rop: mpfr_ptr, op: mpfr_srcptr) -> c_int;
    pub fn mpfr_ceil(rop: mpfr_ptr, op: mpfr_srcptr) -> c_int;
    pub fn mpfr_round(rop: mpfr_ptr, op: mpfr_srcptr) -> c_int;

    // Functions
    pub fn mpfr_sqrt(rop: mpfr_ptr, op: mpfr_srcptr, rnd: mpfr_rnd_t) -> c_int;
    pub fn mpfr_cbrt(rop: mpfr_ptr, op: mpfr_srcptr, rnd: mpfr_rnd_t) -> c_int;
    pub fn mpfr_root(rop: mpfr_ptr, op: mpfr_srcptr, k: c_ulong, rnd: mpfr_rnd_t) -> c_int;
    pub fn mpfr_pow(rop: mpfr_ptr, op1: mpfr_srcptr, op2: mpfr_srcptr, rnd: mpfr_rnd_t) -> c_int;
    pub fn mpfr_abs(rop: mpfr_ptr, op: mpfr_srcptr, rnd: mpfr_rnd_t) -> c_int;
    pub fn mpfr_exp(rop: mpfr_ptr, op: mpfr_srcptr, rnd: mpfr_rnd_t) -> c_int;
    pub fn mpfr_log(rop: mpfr_ptr, op: mpfr_srcptr, rnd: mpfr_rnd_t) -> c_int;
    pub fn mpfr_gamma(rop: mpfr_ptr, op: mpfr_srcptr, rnd: mpfr_rnd_t) -> c_int;
    pub fn mpfr_lngamma(rop: mpfr_ptr, op: mpfr_srcptr, rnd: mpfr_rnd_t) -> c_int;
    pub fn mpfr_lgamma(rop: mpfr_ptr, op: mpfr_srcptr, rnd: mpfr_rnd_t) -> c_int;

    // Formatted output
    pub fn mpfr_snprintf(
        buffer: *const c_char,
        length: size_t,
        string: *const c_char,
        ...
    ) -> c_int;

    #[link_name = "wrapped_mpfr_nan_p"]
    pub fn mpfr_nan_p(x: mpfr_srcptr) -> c_int;
    #[link_name = "wrapped_mpfr_inf_p"]
    pub fn mpfr_inf_p(x: mpfr_srcptr) -> c_int;
    #[link_name = "wrapped_mpfr_zero_p"]
    pub fn mpfr_zero_p(x: mpfr_srcptr) -> c_int;
}

pub struct Mpfr {
    pub mpfr: mpfr_struct,
}

impl fmt::Debug for Mpfr {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&self, fmt)
    }
}

impl fmt::Display for Mpfr {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        if let Ok(s) = self.to_fmt_string("%.Re") {
            fmt.write_str(&s)
        } else {
            Err(fmt::Error)
        }
    }
}

unsafe impl Send for Mpfr {}
unsafe impl Sync for Mpfr {}

impl Drop for Mpfr {
    fn drop(&mut self) {
        unsafe { mpfr_clear(self.as_raw_mut()) }
    }
}

impl Clone for Mpfr {
    fn clone(&self) -> Mpfr {
        let mut mpfr = Mpfr::new2(self.get_prec());
        mpfr.set(self);
        mpfr
    }
}

impl Mpfr {
    #[inline]
    pub fn as_raw(&self) -> &mpfr_struct {
        &self.mpfr
    }

    #[inline]
    pub fn as_raw_mut(&mut self) -> &mut mpfr_struct {
        &mut self.mpfr
    }

    pub fn new() -> Mpfr {
        unsafe {
            let mut mpfr = uninitialized();
            mpfr_init(&mut mpfr);
            Mpfr { mpfr: mpfr }
        }
    }

    pub fn new2(precision: usize) -> Mpfr {
        unsafe {
            let mut mpfr = uninitialized();
            mpfr_init2(&mut mpfr, precision as mpfr_prec_t);
            Mpfr { mpfr: mpfr }
        }
    }

    pub fn new_from_str<T: Into<Vec<u8>>>(s: T, base: usize) -> Option<Mpfr> {
        let c_string = match CString::new(s) {
            Ok(c_string) => c_string,
            Err(..) => return None,
        };
        unsafe {
            let mut mpfr = Mpfr::new();
            if mpfr_set_str(
                mpfr.as_raw_mut(),
                c_string.as_ptr(),
                base as c_int,
                DEFAULT_RND,
            ) == 0
            {
                Some(mpfr)
            } else {
                None
            }
        }
    }

    pub fn new2_from_str<T: Into<Vec<u8>>>(precision: usize, s: T, base: usize) -> Option<Mpfr> {
        let c_string = match CString::new(s) {
            Ok(c_string) => c_string,
            Err(..) => return None,
        };
        unsafe {
            let mut mpfr = Mpfr::new2(precision);
            if mpfr_set_str(
                mpfr.as_raw_mut(),
                c_string.as_ptr(),
                base as c_int,
                DEFAULT_RND,
            ) == 0
            {
                Some(mpfr)
            } else {
                None
            }
        }
    }

    pub fn set(&mut self, other: &Mpfr) {
        unsafe {
            mpfr_set(self.as_raw_mut(), other.as_raw(), DEFAULT_RND);
        }
    }

    pub fn new_u64_2exp(base: u64, exp: i32) -> Mpfr {
        unsafe {
            let mut mpfr = Mpfr::new();
            mpfr_set_ui_2exp(
                mpfr.as_raw_mut(),
                base as c_ulong,
                exp as mpfr_exp_t,
                DEFAULT_RND,
            );
            mpfr
        }
    }

    pub fn new_i64_2exp(base: i64, exp: i32) -> Mpfr {
        unsafe {
            let mut mpfr = Mpfr::new();
            mpfr_set_si_2exp(
                mpfr.as_raw_mut(),
                base as c_long,
                exp as mpfr_exp_t,
                DEFAULT_RND,
            );
            mpfr
        }
    }

    pub fn new_mpz_2exp(base: &Mpz, exp: i32) -> Mpfr {
        unsafe {
            let mut mpfr = Mpfr::new();
            mpfr_set_z_2exp(
                mpfr.as_raw_mut(),
                base.inner(),
                exp as mpfr_exp_t,
                DEFAULT_RND,
            );
            mpfr
        }
    }

    pub fn zero(sign: i32) -> Mpfr {
        unsafe {
            let mut mpfr = Mpfr::new();
            mpfr_set_zero(mpfr.as_raw_mut(), sign as c_int);
            mpfr
        }
    }

    pub fn inf(sign: i32) -> Mpfr {
        unsafe {
            let mut mpfr = Mpfr::new();
            mpfr_set_inf(mpfr.as_raw_mut(), sign as c_int);
            mpfr
        }
    }

    pub fn nan() -> Mpfr {
        unsafe {
            let mut mpfr = Mpfr::new();
            mpfr_set_nan(mpfr.as_raw_mut());
            mpfr
        }
    }

    pub fn get_default_prec() -> usize {
        unsafe { mpfr_get_default_prec() as usize }
    }

    pub fn set_default_prec(precision: usize) {
        unsafe {
            mpfr_set_default_prec(precision as mpfr_prec_t);
        }
    }

    pub fn get_prec(&self) -> usize {
        unsafe { mpfr_get_prec(self.as_raw()) as usize }
    }

    pub fn get_exp(&self) -> i64 {
        unsafe { mpfr_get_exp(self.as_raw()) as _ }
    }

    pub fn set_prec(&mut self, precision: usize) {
        unsafe {
            mpfr_set_prec(self.as_raw_mut(), precision as mpfr_prec_t);
        }
    }

    // Rounding

    pub fn floor(&self) -> Mpfr {
        unsafe {
            let mut res = Mpfr::new2(self.get_prec());
            mpfr_floor(res.as_raw_mut(), self.as_raw());
            res
        }
    }

    pub fn ceil(&self) -> Mpfr {
        unsafe {
            let mut res = Mpfr::new2(self.get_prec());
            mpfr_ceil(res.as_raw_mut(), self.as_raw());
            res
        }
    }

    pub fn round(&self) -> Mpfr {
        unsafe {
            let mut res = Mpfr::new2(self.get_prec());
            mpfr_round(res.as_raw_mut(), self.as_raw());
            res
        }
    }

    // Mathematical functions

    pub fn sqrt(&self) -> Mpfr {
        unsafe {
            let mut res = Mpfr::new2(self.get_prec());
            mpfr_sqrt(res.as_raw_mut(), self.as_raw(), DEFAULT_RND);
            res
        }
    }

    pub fn cbrt(&self) -> Mpfr {
        unsafe {
            let mut res = Mpfr::new2(self.get_prec());
            mpfr_cbrt(res.as_raw_mut(), self.as_raw(), DEFAULT_RND);
            res
        }
    }

    pub fn root(&self, k: u64) -> Mpfr {
        unsafe {
            let mut res = Mpfr::new2(self.get_prec());
            mpfr_root(res.as_raw_mut(), self.as_raw(), k as c_ulong, DEFAULT_RND);
            res
        }
    }

    pub fn pow(&self, other: &Mpfr) -> Mpfr {
        unsafe {
            let mut res = Mpfr::new2(self.get_prec());
            mpfr_pow(res.as_raw_mut(), self.as_raw(), other.as_raw(), DEFAULT_RND);
            res
        }
    }

    pub fn abs(&self) -> Mpfr {
        unsafe {
            let mut res = Mpfr::new2(self.get_prec());
            mpfr_abs(res.as_raw_mut(), self.as_raw(), DEFAULT_RND);
            res
        }
    }

    pub fn exp(&self) -> Mpfr {
        unsafe {
            let mut res = Mpfr::new2(self.get_prec());
            mpfr_exp(res.as_raw_mut(), self.as_raw(), DEFAULT_RND);
            res
        }
    }

    pub fn log(&self) -> Mpfr {
        unsafe {
            let mut res = Mpfr::new2(self.get_prec());
            mpfr_log(res.as_raw_mut(), self.as_raw(), DEFAULT_RND);
            res
        }
    }

    pub fn gamma(&self) -> Mpfr {
        unsafe {
            let mut res = Mpfr::new2(self.get_prec());
            mpfr_gamma(res.as_raw_mut(), self.as_raw(), DEFAULT_RND);
            res
        }
    }

    pub fn lngamma(&self) -> Mpfr {
        unsafe {
            let mut res = Mpfr::new2(self.get_prec());
            mpfr_lngamma(res.as_raw_mut(), self.as_raw(), DEFAULT_RND);
            res
        }
    }

    pub fn lgamma(&self) -> Mpfr {
        unsafe {
            let mut res = Mpfr::new2(self.get_prec());
            mpfr_lgamma(res.as_raw_mut(), self.as_raw(), DEFAULT_RND);
            res
        }
    }

    pub fn to_fmt_string(&self, template: &str) -> Result<String, FmtStringError> {
        unsafe {
            if let Ok(template) = CString::new(template) {
                let length = mpfr_snprintf(ptr::null(), 0, template.as_ptr(), self.as_raw());
                if length < 0 {
                    return Err(FmtStringError);
                }
                let buff: Vec<c_char> = Vec::with_capacity((length + 1) as usize);
                mpfr_snprintf(
                    buff.as_ptr(),
                    (length + 1) as size_t,
                    template.as_ptr(),
                    self.as_raw(),
                );
                let s = CStr::from_ptr(buff.as_ptr());
                match s.to_str() {
                    Ok(s) => Ok(s.to_string()),
                    _ => Err(FmtStringError),
                }
            } else {
                Err(FmtStringError)
            }
        }
    }

    #[inline]
    pub fn is_nan(&self) -> bool {
        unsafe { mpfr_nan_p(self.as_raw()) != 0 }
    }

    #[inline]
    pub fn is_zero(&self) -> bool {
        unsafe { mpfr_zero_p(self.as_raw()) != 0 }
    }

    #[inline]
    pub fn is_infinity(&self) -> bool {
        unsafe { mpfr_inf_p(self.as_raw()) != 0 }
    }

    impl_mut_c_wrapper_w_default_rnd!(
        add_mut,
        mpfr_add,
        (x: SelfRef, y: SelfRef),
        doc = "`self = x + y`"
    );

    impl_mut_c_wrapper_w_default_rnd!(
        sub_mut,
        mpfr_sub,
        (x: SelfRef, y: SelfRef),
        doc = "`self = x - y`"
    );

    impl_mut_c_wrapper_w_default_rnd!(
        mul_mut,
        mpfr_mul,
        (x: SelfRef, y: SelfRef),
        doc = "`self = x * y`"
    );

    impl_mut_c_wrapper_w_default_rnd!(
        div_mut,
        mpfr_div,
        (x: SelfRef, y: SelfRef),
        doc = "`self = x / y`"
    );
}

pub struct FmtStringError;

impl PartialEq for Mpfr {
    fn eq(&self, other: &Mpfr) -> bool {
        if self.is_nan() || other.is_nan() {
            false
        } else {
            unsafe { mpfr_cmp(self.as_raw(), other.as_raw()) == 0 }
        }
    }
}

impl PartialOrd for Mpfr {
    fn partial_cmp(&self, other: &Mpfr) -> Option<Ordering> {
        if self.is_nan() || other.is_nan() {
            None
        } else {
            Some(int_to_ord!(unsafe {
                mpfr_cmp(self.as_raw(), other.as_raw())
            }))
        }
    }
}

// Conversions

impl From<i64> for Mpfr {
    fn from(x: i64) -> Mpfr {
        unsafe {
            let mut mpfr = Mpfr::new();
            mpfr_set_si(mpfr.as_raw_mut(), x as c_long, DEFAULT_RND);
            mpfr
        }
    }
}

impl From<u64> for Mpfr {
    fn from(x: u64) -> Mpfr {
        unsafe {
            let mut mpfr = Mpfr::new();
            mpfr_set_ui(mpfr.as_raw_mut(), x as c_ulong, DEFAULT_RND);
            mpfr
        }
    }
}

impl From<f64> for Mpfr {
    fn from(x: f64) -> Mpfr {
        unsafe {
            let mut mpfr = Mpfr::new();
            mpfr_set_d(mpfr.as_raw_mut(), x as c_double, DEFAULT_RND);
            mpfr
        }
    }
}

impl From<Mpz> for Mpfr {
    fn from(x: Mpz) -> Mpfr {
        unsafe {
            let mut mpfr = Mpfr::new();
            mpfr_set_z(mpfr.as_raw_mut(), x.inner(), DEFAULT_RND);
            mpfr
        }
    }
}

impl From<Mpq> for Mpfr {
    fn from(x: Mpq) -> Mpfr {
        unsafe {
            let mut mpfr = Mpfr::new();
            mpfr_set_q(mpfr.as_raw_mut(), x.inner(), DEFAULT_RND);
            mpfr
        }
    }
}

impl From<Mpf> for Mpfr {
    fn from(x: Mpf) -> Mpfr {
        unsafe {
            let mut mpfr = Mpfr::new();
            mpfr_set_f(mpfr.as_raw_mut(), x.inner(), DEFAULT_RND);
            mpfr
        }
    }
}

impl<'a> Into<i64> for &'a Mpfr {
    fn into(self) -> i64 {
        unsafe { mpfr_get_si(self.as_raw(), DEFAULT_RND) as i64 }
    }
}

impl<'a> Into<u64> for &'a Mpfr {
    fn into(self) -> u64 {
        unsafe { mpfr_get_ui(self.as_raw(), DEFAULT_RND) as u64 }
    }
}

impl<'a> Into<f64> for &'a Mpfr {
    fn into(self) -> f64 {
        unsafe { mpfr_get_d(self.as_raw(), DEFAULT_RND) as f64 }
    }
}

impl<'a> Into<Mpz> for &'a Mpfr {
    fn into(self) -> Mpz {
        unsafe {
            let mut result = Mpz::new();
            mpfr_get_z(result.inner_mut(), self.as_raw(), DEFAULT_RND);
            result
        }
    }
}

impl<'a> Into<Mpf> for &'a Mpfr {
    fn into(self) -> Mpf {
        unsafe {
            let mut result = Mpf::new(self.get_prec());
            mpfr_get_f(result.inner_mut(), self.as_raw(), DEFAULT_RND);
            result
        }
    }
}

//
// Addition
//
// Supports:
// Mpfr + Mpfr
// Mpfr + f64, f64 + Mpfr
// Mpfr + i64, i64 + Mpfr
//
//

impl<'a, 'b> Add<&'a Mpfr> for &'b Mpfr {
    type Output = Mpfr;
    fn add(self, other: &Mpfr) -> Mpfr {
        unsafe {
            let mut res = Mpfr::new2(cmp::max(self.get_prec(), other.get_prec()));
            mpfr_add(res.as_raw_mut(), self.as_raw(), other.as_raw(), DEFAULT_RND);
            res
        }
    }
}

impl<'a> Add<&'a Mpfr> for Mpfr {
    type Output = Mpfr;
    #[inline]
    fn add(mut self, other: &Mpfr) -> Mpfr {
        if other.get_prec() > self.get_prec() {
            return &self + other;
        }
        unsafe {
            mpfr_add(
                self.as_raw_mut(),
                self.as_raw(),
                other.as_raw(),
                DEFAULT_RND,
            );
            self
        }
    }
}

impl Add<Mpfr> for f64 {
    type Output = Mpfr;
    fn add(self, other: Mpfr) -> Mpfr {
        unsafe {
            let mut res = Mpfr::new2(other.get_prec());
            mpfr_add_d(
                res.as_raw_mut(),
                other.as_raw(),
                self as c_double,
                DEFAULT_RND,
            );
            res
        }
    }
}

impl<'a> Add<&'a Mpfr> for f64 {
    type Output = Mpfr;
    fn add(self, other: &'a Mpfr) -> Mpfr {
        unsafe {
            let mut res = Mpfr::new2(other.get_prec());
            mpfr_add_d(
                res.as_raw_mut(),
                other.as_raw(),
                self as c_double,
                DEFAULT_RND,
            );
            res
        }
    }
}

impl Add<f64> for Mpfr {
    type Output = Mpfr;
    #[inline]
    fn add(mut self, other: f64) -> Mpfr {
        unsafe {
            mpfr_add_d(
                self.as_raw_mut(),
                self.as_raw(),
                other as c_double,
                DEFAULT_RND,
            );
            self
        }
    }
}

impl<'a> Add<f64> for &'a Mpfr {
    type Output = Mpfr;
    fn add(self, other: f64) -> Mpfr {
        unsafe {
            let mut res = Mpfr::new2(self.get_prec());
            mpfr_add_d(
                res.as_raw_mut(),
                self.as_raw(),
                other as c_double,
                DEFAULT_RND,
            );
            res
        }
    }
}

impl Add<Mpfr> for i64 {
    type Output = Mpfr;
    fn add(self, other: Mpfr) -> Mpfr {
        unsafe {
            let mut res = Mpfr::new2(other.get_prec());
            mpfr_add_si(
                res.as_raw_mut(),
                other.as_raw(),
                self as c_long,
                DEFAULT_RND,
            );
            res
        }
    }
}

impl<'a> Add<&'a Mpfr> for i64 {
    type Output = Mpfr;
    fn add(self, other: &'a Mpfr) -> Mpfr {
        unsafe {
            let mut res = Mpfr::new2(other.get_prec());
            mpfr_add_si(
                res.as_raw_mut(),
                other.as_raw(),
                self as c_long,
                DEFAULT_RND,
            );
            res
        }
    }
}

impl Add<i64> for Mpfr {
    type Output = Mpfr;
    #[inline]
    fn add(mut self, other: i64) -> Mpfr {
        unsafe {
            mpfr_add_si(
                self.as_raw_mut(),
                self.as_raw(),
                other as c_long,
                DEFAULT_RND,
            );
            self
        }
    }
}

impl<'a> Add<i64> for &'a Mpfr {
    type Output = Mpfr;
    fn add(self, other: i64) -> Mpfr {
        unsafe {
            let mut res = Mpfr::new2(self.get_prec());
            mpfr_add_si(
                res.as_raw_mut(),
                self.as_raw(),
                other as c_long,
                DEFAULT_RND,
            );
            res
        }
    }
}

//
// Subtraction
//
// Supports:
// Mpfr - Mpfr
// Mpfr - f64, f64 - Mpfr
// Mpfr - i64, i64 - Mpfr
//
//

impl<'a, 'b> Sub<&'a Mpfr> for &'b Mpfr {
    type Output = Mpfr;
    fn sub(self, other: &Mpfr) -> Mpfr {
        unsafe {
            let mut res = Mpfr::new2(cmp::max(self.get_prec(), other.get_prec()));
            mpfr_sub(res.as_raw_mut(), self.as_raw(), other.as_raw(), DEFAULT_RND);
            res
        }
    }
}

impl<'a> Sub<&'a Mpfr> for Mpfr {
    type Output = Mpfr;
    #[inline]
    fn sub(mut self, other: &Mpfr) -> Mpfr {
        if other.get_prec() > self.get_prec() {
            return &self - other;
        }
        unsafe {
            mpfr_sub(
                self.as_raw_mut(),
                self.as_raw(),
                other.as_raw(),
                DEFAULT_RND,
            );
            self
        }
    }
}

impl Sub<Mpfr> for f64 {
    type Output = Mpfr;
    fn sub(self, other: Mpfr) -> Mpfr {
        unsafe {
            let mut res = Mpfr::new2(other.get_prec());
            mpfr_d_sub(
                res.as_raw_mut(),
                self as c_double,
                other.as_raw(),
                DEFAULT_RND,
            );
            res
        }
    }
}

impl<'a> Sub<&'a Mpfr> for f64 {
    type Output = Mpfr;
    fn sub(self, other: &'a Mpfr) -> Mpfr {
        unsafe {
            let mut res = Mpfr::new2(other.get_prec());
            mpfr_d_sub(
                res.as_raw_mut(),
                self as c_double,
                other.as_raw(),
                DEFAULT_RND,
            );
            res
        }
    }
}

impl Sub<f64> for Mpfr {
    type Output = Mpfr;
    #[inline]
    fn sub(mut self, other: f64) -> Mpfr {
        unsafe {
            mpfr_sub_d(
                self.as_raw_mut(),
                self.as_raw(),
                other as c_double,
                DEFAULT_RND,
            );
            self
        }
    }
}

impl<'a> Sub<f64> for &'a Mpfr {
    type Output = Mpfr;
    fn sub(self, other: f64) -> Mpfr {
        unsafe {
            let mut res = Mpfr::new2(self.get_prec());
            mpfr_sub_d(
                res.as_raw_mut(),
                self.as_raw(),
                other as c_double,
                DEFAULT_RND,
            );
            res
        }
    }
}

impl Sub<Mpfr> for i64 {
    type Output = Mpfr;
    fn sub(self, other: Mpfr) -> Mpfr {
        unsafe {
            let mut res = Mpfr::new2(other.get_prec());
            mpfr_si_sub(
                res.as_raw_mut(),
                self as c_long,
                other.as_raw(),
                DEFAULT_RND,
            );
            res
        }
    }
}

impl<'a> Sub<&'a Mpfr> for i64 {
    type Output = Mpfr;
    fn sub(self, other: &'a Mpfr) -> Mpfr {
        unsafe {
            let mut res = Mpfr::new2(other.get_prec());
            mpfr_si_sub(
                res.as_raw_mut(),
                self as c_long,
                other.as_raw(),
                DEFAULT_RND,
            );
            res
        }
    }
}

impl Sub<i64> for Mpfr {
    type Output = Mpfr;
    #[inline]
    fn sub(mut self, other: i64) -> Mpfr {
        unsafe {
            mpfr_sub_si(
                self.as_raw_mut(),
                self.as_raw(),
                other as c_long,
                DEFAULT_RND,
            );
            self
        }
    }
}

impl<'a> Sub<i64> for &'a Mpfr {
    type Output = Mpfr;
    fn sub(self, other: i64) -> Mpfr {
        unsafe {
            let mut res = Mpfr::new2(self.get_prec());
            mpfr_sub_si(
                res.as_raw_mut(),
                self.as_raw(),
                other as c_long,
                DEFAULT_RND,
            );
            res
        }
    }
}

//
// Multiplication
//
// Supports:
// Mpfr * Mpfr
// Mpfr * f64, f64 * Mpfr
// Mpfr * i64, i64 * Mpfr
//
//

impl<'a, 'b> Mul<&'a Mpfr> for &'b Mpfr {
    type Output = Mpfr;
    fn mul(self, other: &Mpfr) -> Mpfr {
        unsafe {
            let mut res = Mpfr::new2(cmp::max(self.get_prec(), other.get_prec()));
            mpfr_mul(res.as_raw_mut(), self.as_raw(), other.as_raw(), DEFAULT_RND);
            res
        }
    }
}

impl<'a> Mul<&'a Mpfr> for Mpfr {
    type Output = Mpfr;

    #[inline]
    fn mul(mut self, other: &Mpfr) -> Mpfr {
        if other.get_prec() > self.get_prec() {
            return &self * other;
        }
        unsafe {
            mpfr_mul(
                self.as_raw_mut(),
                self.as_raw(),
                other.as_raw(),
                DEFAULT_RND,
            );
            self
        }
    }
}

impl Mul<Mpfr> for f64 {
    type Output = Mpfr;
    fn mul(self, other: Mpfr) -> Mpfr {
        unsafe {
            let mut res = Mpfr::new2(other.get_prec());
            mpfr_mul_d(
                res.as_raw_mut(),
                other.as_raw(),
                self as c_double,
                DEFAULT_RND,
            );
            res
        }
    }
}

impl<'a> Mul<&'a Mpfr> for f64 {
    type Output = Mpfr;
    fn mul(self, other: &'a Mpfr) -> Mpfr {
        unsafe {
            let mut res = Mpfr::new2(other.get_prec());
            mpfr_mul_d(
                res.as_raw_mut(),
                other.as_raw(),
                self as c_double,
                DEFAULT_RND,
            );
            res
        }
    }
}

impl Mul<f64> for Mpfr {
    type Output = Mpfr;
    #[inline]
    fn mul(mut self, other: f64) -> Mpfr {
        unsafe {
            mpfr_mul_d(
                self.as_raw_mut(),
                self.as_raw(),
                other as c_double,
                DEFAULT_RND,
            );
            self
        }
    }
}

impl<'a> Mul<f64> for &'a Mpfr {
    type Output = Mpfr;
    fn mul(self, other: f64) -> Mpfr {
        unsafe {
            let mut res = Mpfr::new2(self.get_prec());
            mpfr_mul_d(
                res.as_raw_mut(),
                self.as_raw(),
                other as c_double,
                DEFAULT_RND,
            );
            res
        }
    }
}

impl Mul<Mpfr> for i64 {
    type Output = Mpfr;
    fn mul(self, other: Mpfr) -> Mpfr {
        unsafe {
            let mut res = Mpfr::new2(other.get_prec());
            mpfr_mul_si(
                res.as_raw_mut(),
                other.as_raw(),
                self as c_long,
                DEFAULT_RND,
            );
            res
        }
    }
}

impl<'a> Mul<&'a Mpfr> for i64 {
    type Output = Mpfr;
    fn mul(self, other: &'a Mpfr) -> Mpfr {
        unsafe {
            let mut res = Mpfr::new2(other.get_prec());
            mpfr_mul_si(
                res.as_raw_mut(),
                other.as_raw(),
                self as c_long,
                DEFAULT_RND,
            );
            res
        }
    }
}

impl Mul<i64> for Mpfr {
    type Output = Mpfr;
    #[inline]
    fn mul(mut self, other: i64) -> Mpfr {
        unsafe {
            mpfr_mul_si(
                self.as_raw_mut(),
                self.as_raw(),
                other as c_long,
                DEFAULT_RND,
            );
            self
        }
    }
}

impl<'a> Mul<i64> for &'a Mpfr {
    type Output = Mpfr;
    fn mul(self, other: i64) -> Mpfr {
        unsafe {
            let mut res = Mpfr::new2(self.get_prec());
            mpfr_mul_si(
                res.as_raw_mut(),
                self.as_raw(),
                other as c_long,
                DEFAULT_RND,
            );
            res
        }
    }
}

//
// Division
//
// Supports:
// Mpfr / Mpfr
// Mpfr / f64, f64 / Mpfr
// Mpfr / i64, i64 / Mpfr
//
//

macro_rules! div_by_zero_check {
    ($expr: expr) => {
        if $expr.is_zero() {
            panic!("divide by zero")
        }
    }
}

impl<'a, 'b> Div<&'a Mpfr> for &'b Mpfr {
    type Output = Mpfr;
    fn div(self, other: &Mpfr) -> Mpfr {
        unsafe {
            div_by_zero_check!(other);
            let mut res = Mpfr::new2(cmp::max(self.get_prec(), other.get_prec()));
            mpfr_div(res.as_raw_mut(), self.as_raw(), other.as_raw(), DEFAULT_RND);
            res
        }
    }
}

impl<'a> Div<&'a Mpfr> for Mpfr {
    type Output = Mpfr;
    #[inline]
    fn div(mut self, other: &Mpfr) -> Mpfr {
        if other.get_prec() > self.get_prec() {
            return &self / other;
        }
        unsafe {
            div_by_zero_check!(other);
            mpfr_div(
                self.as_raw_mut(),
                self.as_raw(),
                other.as_raw(),
                DEFAULT_RND,
            );
            self
        }
    }
}

impl Div<Mpfr> for f64 {
    type Output = Mpfr;
    fn div(self, other: Mpfr) -> Mpfr {
        unsafe {
            div_by_zero_check!(other);

            let mut res = Mpfr::new2(other.get_prec());
            mpfr_d_div(
                res.as_raw_mut(),
                self as c_double,
                other.as_raw(),
                DEFAULT_RND,
            );
            res
        }
    }
}

impl<'a> Div<&'a Mpfr> for f64 {
    type Output = Mpfr;
    fn div(self, other: &'a Mpfr) -> Mpfr {
        unsafe {
            div_by_zero_check!(other);

            let mut res = Mpfr::new2(other.get_prec());
            mpfr_d_div(
                res.as_raw_mut(),
                self as c_double,
                other.as_raw(),
                DEFAULT_RND,
            );
            res
        }
    }
}

impl Div<f64> for Mpfr {
    type Output = Mpfr;
    #[inline]
    fn div(mut self, other: f64) -> Mpfr {
        unsafe {
            if other == 0.0 {
                panic!("divide by zero")
            }

            mpfr_div_d(
                self.as_raw_mut(),
                self.as_raw(),
                other as c_double,
                DEFAULT_RND,
            );
            self
        }
    }
}

impl<'a> Div<f64> for &'a Mpfr {
    type Output = Mpfr;
    fn div(self, other: f64) -> Mpfr {
        unsafe {
            if other == 0.0 {
                panic!("divide by zero")
            }

            let mut res = Mpfr::new2(self.get_prec());
            mpfr_div_d(
                res.as_raw_mut(),
                self.as_raw(),
                other as c_double,
                DEFAULT_RND,
            );
            res
        }
    }
}

impl Div<Mpfr> for i64 {
    type Output = Mpfr;
    fn div(self, other: Mpfr) -> Mpfr {
        unsafe {
            div_by_zero_check!(other);

            let mut res = Mpfr::new2(other.get_prec());
            mpfr_si_div(
                res.as_raw_mut(),
                self as c_long,
                other.as_raw(),
                DEFAULT_RND,
            );
            res
        }
    }
}

impl<'a> Div<&'a Mpfr> for i64 {
    type Output = Mpfr;
    fn div(self, other: &'a Mpfr) -> Mpfr {
        unsafe {
            div_by_zero_check!(other);

            let mut res = Mpfr::new2(other.get_prec());
            mpfr_si_div(
                res.as_raw_mut(),
                self as c_long,
                other.as_raw(),
                DEFAULT_RND,
            );
            res
        }
    }
}

impl Div<i64> for Mpfr {
    type Output = Mpfr;
    #[inline]
    fn div(mut self, other: i64) -> Mpfr {
        unsafe {
            if other == 0 {
                panic!("divide by zero")
            }

            mpfr_div_si(
                self.as_raw_mut(),
                self.as_raw(),
                other as c_long,
                DEFAULT_RND,
            );
            self
        }
    }
}

impl<'a> Div<i64> for &'a Mpfr {
    type Output = Mpfr;
    fn div(self, other: i64) -> Mpfr {
        unsafe {
            if other == 0 {
                panic!("divide by zero")
            }

            let mut res = Mpfr::new2(self.get_prec());
            mpfr_div_si(
                res.as_raw_mut(),
                self.as_raw(),
                other as c_long,
                DEFAULT_RND,
            );
            res
        }
    }
}

// Negation

impl<'b> Neg for &'b Mpfr {
    type Output = Mpfr;
    fn neg(self) -> Mpfr {
        unsafe {
            let mut res = Mpfr::new2(self.get_prec());
            mpfr_neg(res.as_raw_mut(), self.as_raw(), DEFAULT_RND);
            res
        }
    }
}

impl Neg for Mpfr {
    type Output = Mpfr;
    #[inline]
    fn neg(mut self) -> Mpfr {
        unsafe {
            mpfr_neg(self.as_raw_mut(), self.as_raw(), DEFAULT_RND);
            self
        }
    }
}

gen_overloads!(Mpfr);

define_assign_c!(Mpfr, AddAssign, add_assign, mpfr_add_si, c_long);
define_assign_c!(Mpfr, AddAssign, add_assign, mpfr_add_d, c_double);
define_assign_c!(Mpfr, SubAssign, sub_assign, mpfr_sub_si, c_long);
define_assign_c!(Mpfr, SubAssign, sub_assign, mpfr_sub_d, c_double);
define_assign_c!(Mpfr, MulAssign, mul_assign, mpfr_mul_si, c_long);
define_assign_c!(Mpfr, MulAssign, mul_assign, mpfr_mul_d, c_double);
define_assign_c!(Mpfr, DivAssign, div_assign, mpfr_div_si, c_long);
define_assign_c!(Mpfr, DivAssign, div_assign, mpfr_div_d, c_double);

define_assign_wref!(Mpfr, AddAssign, add_assign, mpfr_add, Mpfr);
define_assign_wref!(Mpfr, SubAssign, sub_assign, mpfr_sub, Mpfr);
define_assign_wref!(Mpfr, MulAssign, mul_assign, mpfr_mul, Mpfr);
define_assign_wref!(Mpfr, DivAssign, div_assign, mpfr_div, Mpfr);

// serde support

impl Serialize for Mpfr {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        String::serialize(&self.to_string(), serializer)
    }
}

impl<'de> Deserialize<'de> for Mpfr {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        match Mpfr::new_from_str(s, 10) {
            Some(val) => Ok(val),
            None => Err(de::Error::custom("Cannot parse decimal float")),
        }
    }
}
