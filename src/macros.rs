macro_rules! gen_overloads_inner {
    ($tr:ident, $meth:ident, $T:ident) => {
        impl<'a> $tr <$T> for &'a $T {
            type Output = $T;
            #[inline]
            fn $meth(self, other: $T) -> $T {
                self.$meth(&other)
            }
        }
        impl $tr<$T> for $T {
            type Output = $T;
            #[inline]
            fn $meth(self, other: $T) -> $T {
                self.$meth(&other)
            }
        }
    }
}

macro_rules! gen_overloads {
    ($T:ident) => {
        gen_overloads_inner!(Add, add, $T);
        gen_overloads_inner!(Sub, sub, $T);
        gen_overloads_inner!(Mul, mul, $T);
        gen_overloads_inner!(Div, div, $T);
    }
}

macro_rules! int_to_ord {
    ($cmp: expr) => {
        {
            let cmp = $cmp;
            if cmp == 0 {
                Equal
            } else if cmp < 0 {
                Less
            } else {
                Greater
            }
        }
    }
}

macro_rules! __ref_or_val {
    (Si, $val: expr) => {$val};
    (Ui, $val: expr) => {$val};
    (SelfRef, $val: expr) => {$val.as_raw()};
    (SelfRefMut, $val: expr) => {$val.as_raw_mut()};
    ($t: ident, $val: expr) =>  {$val};
}

macro_rules! __ann_type {
    (SelfRef) => {&Self};
    (SelfRefMut) => {&mut Self};
    (Si) => {c_long};
    (Ui) => {c_ulong};
    ($t: ident) => {$t};
}

macro_rules! impl_mut_c_wrapper_w_default_rnd {
    ($meth: ident, $c_func: ident, ($($x:ident: $t:ident),*), $($m: meta),*) => {
        $(#[$m])*
        pub fn $meth(&mut self, $($x: __ann_type!($t)),*) {
            unsafe {
                $c_func(self.as_raw_mut(), $(__ref_or_val!($t, $x)),*, DEFAULT_RND);
            }
        }
    };
}

macro_rules! define_assign_c {
    ($t:ty, $trait:ident, $meth:ident, $func:ident, $typ:ty) =>
    {
        impl $trait<$typ> for $t {
            fn $meth(&mut self, other: $typ) {
                unsafe {
                    $func(self.as_raw_mut(), self.as_raw(), other, DEFAULT_RND);
                }
            }
        }
    }
}

macro_rules! define_assign_wref {
    ($t:ty, $trait:ident, $meth:ident, $func:ident, $ty:ty) =>
    {
        impl<'a> $trait<&'a $ty> for $t {
            fn $meth(&mut self, other: &$ty) {
                unsafe {
                    $func(self.as_raw_mut(), self.as_raw(), other.as_raw(), DEFAULT_RND);
                }
            }
        }
    };
}
