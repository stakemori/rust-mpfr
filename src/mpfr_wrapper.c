#include "mpfr.h"

int wrapped_mpfr_nan_p(mpfr_t x) { return mpfr_nan_p(x); }
int wrapped_mpfr_inf_p(mpfr_t x) { return mpfr_inf_p(x); }
