//! Self-contained f32 math lowering. Transcendentals use PTX SFU approximations;
//! inverse functions use range reduction and a convergent atan series.
use super::*;

impl Emitter<'_> {
    fn f(&mut self, op: &str, args: &[&str]) -> String {
        self.calc(op, "f32", args)
    }

    fn add(&mut self, a: &str, b: &str) -> String {
        self.f("add.rn.f32", &[a, b])
    }

    fn sub(&mut self, a: &str, b: &str) -> String {
        self.f("sub.rn.f32", &[a, b])
    }

    fn mul(&mut self, a: &str, b: &str) -> String {
        self.f("mul.rn.f32", &[a, b])
    }

    fn div(&mut self, a: &str, b: &str) -> String {
        self.f("div.rn.f32", &[a, b])
    }

    fn abs(&mut self, a: &str) -> String {
        self.f("abs.f32", &[a])
    }

    fn pred(&mut self, comparison: &str, a: &str, b: &str) -> String {
        self.calc(&format!("setp.{comparison}.f32"), "pred", &[a, b])
    }

    fn select(&mut self, p: &str, a: &str, b: &str) -> String {
        self.f("selp.f32", &[a, b, p])
    }

    fn sign(&mut self, magnitude: &str, sign: &str) -> String {
        // PTX copysign takes the sign first, magnitude second.
        self.f("copysign.f32", &[sign, magnitude])
    }

    fn negative(&mut self, a: &str) -> String {
        let bits = self.calc("mov.b32", "b32", &[a]);
        let bits = self.calc("and.b32", "b32", &[&bits, "2147483648"]);
        self.calc("setp.ne.u32", "pred", &[&bits, "0"])
    }

    fn exp(&mut self, a: &str) -> String {
        let x = self.mul(a, &float(std::f32::consts::LOG2_E));
        self.f("ex2.approx.f32", &[&x])
    }

    fn log(&mut self, a: &str) -> String {
        let x = self.f("lg2.approx.f32", &[a]);
        self.mul(&x, &float(std::f32::consts::LN_2))
    }

    // Stable near zero: series through x^9, selected only for |x| < 1/8.
    fn expm1(&mut self, x: &str) -> String {
        let mut p = float(1.0 / 362880.0);
        for coefficient in [
            1.0 / 40320.0,
            1.0 / 5040.0,
            1.0 / 720.0,
            1.0 / 120.0,
            1.0 / 24.0,
            1.0 / 6.0,
            0.5,
            1.0,
        ] {
            p = self.f("fma.rn.f32", &[&p, x, &float(coefficient)]);
        }
        let small = self.mul(x, &p);
        let exponential = self.exp(x);
        let large = self.sub(&exponential, &float(1.0));
        let abs = self.abs(x);
        let p = self.pred("lt", &abs, &float(0.125));
        self.select(&p, &small, &large)
    }

    fn log1p(&mut self, x: &str) -> String {
        // log(1+x) = 2*(z + z^3/3 + ...), z=x/(2+x).
        let denom = self.add(x, &float(2.0));
        let z = self.div(x, &denom);
        let z2 = self.mul(&z, &z);
        let mut p = float(1.0 / 13.0);
        for n in [11, 9, 7, 5, 3, 1] {
            p = self.f("fma.rn.f32", &[&p, &z2, &float(1.0 / n as f32)]);
        }
        let p = self.mul(&p, &z);
        let small = self.mul(&p, &float(2.0));
        let one_plus = self.add(x, &float(1.0));
        let large = self.log(&one_plus);
        let abs = self.abs(x);
        let p = self.pred("lt", &abs, &float(0.5));
        let result = self.select(&p, &small, &large);
        let tiny = self.pred("lt", &abs, &float(1e-8));
        self.select(&tiny, x, &result)
    }

    fn atan(&mut self, x: &str) -> String {
        let abs = self.abs(x);
        let reciprocal = self.div(&float(1.0), &abs);
        let invert = self.pred("gt", &abs, &float(1.0));
        let reduced = self.select(&invert, &reciprocal, &abs);
        let numerator = self.sub(&reduced, &float(1.0));
        let denominator = self.add(&reduced, &float(1.0));
        let shifted = self.div(&numerator, &denominator);
        let shift = self.pred("gt", &reduced, &float(std::f32::consts::SQRT_2 - 1.0));
        let z = self.select(&shift, &shifted, &reduced);
        let z2 = self.mul(&z, &z);
        // |z| <= tan(pi/8). Alternating series truncation < 3e-11.
        let mut p = float(-1.0 / 23.0);
        for n in (0..11).rev() {
            let coefficient = if n % 2 == 0 { 1.0 } else { -1.0 } / (2 * n + 1) as f32;
            p = self.f("fma.rn.f32", &[&p, &z2, &float(coefficient)]);
        }
        let small = self.mul(&z, &p);
        let shifted = self.add(&small, &float(std::f32::consts::FRAC_PI_4));
        let angle = self.select(&shift, &shifted, &small);
        let inverse = self.sub(&float(std::f32::consts::FRAC_PI_2), &angle);
        let angle = self.select(&invert, &inverse, &angle);
        self.sign(&angle, x)
    }
    pub(super) fn atan2(&mut self, y: &str, x: &str) -> String {
        let ax = self.abs(x);
        let ay = self.abs(y);
        let high = self.pred("gt", &ay, &ax);
        let lo = self.select(&high, &ax, &ay);
        let hi = self.select(&high, &ay, &ax);
        let ratio = self.div(&lo, &hi);
        let both_inf = self.pred("eq", &lo, &float(f32::INFINITY));
        let ratio = self.select(&both_inf, &float(1.0), &ratio);
        let both_zero = self.pred("eq", &hi, &float(0.0));
        let ratio = self.select(&both_zero, &float(0.0), &ratio);
        let angle = self.atan(&ratio);
        let inverted = self.sub(&float(std::f32::consts::FRAC_PI_2), &angle);
        let angle = self.select(&high, &inverted, &angle);
        let negative_x = self.negative(x);
        let quadrant = self.sub(&float(std::f32::consts::PI), &angle);
        let angle = self.select(&negative_x, &quadrant, &angle);
        let angle = self.sign(&angle, y);
        let nan = self.pred("nan", x, y);
        self.select(&nan, &float(f32::NAN), &angle)
    }
    pub(super) fn unary_math(&mut self, op: UnaryIntrinsic, x: &str) -> String {
        use UnaryIntrinsic::*;
        match op {
            Sin => self.f("sin.approx.f32", &[x]),
            Cos => self.f("cos.approx.f32", &[x]),
            Tan => {
                let sin = self.f("sin.approx.f32", &[x]);
                let cos = self.f("cos.approx.f32", &[x]);
                self.div(&sin, &cos)
            }
            Atan => self.atan(x),
            Asin | Acos => {
                let a = self.sub(&float(1.0), x);
                let b = self.add(&float(1.0), x);
                let radicand = self.mul(&a, &b);
                let root = self.f("sqrt.rn.f32", &[&radicand]);
                if op == Asin {
                    self.atan2(x, &root)
                } else {
                    self.atan2(&root, x)
                }
            }
            Exp => self.exp(x),
            Log => self.log(x),
            Exp2 => self.f("ex2.approx.f32", &[x]),
            Log2 => self.f("lg2.approx.f32", &[x]),
            Sqrt => self.f("sqrt.rn.f32", &[x]),
            InverseSqrt => {
                let root = self.f("sqrt.rn.f32", &[x]);
                self.div(&float(1.0), &root)
            }
            Floor => self.f("cvt.rmi.f32.f32", &[x]),
            Ceil => self.f("cvt.rpi.f32.f32", &[x]),
            Sinh | Cosh => {
                let abs = self.abs(x);
                let normal = if op == Sinh {
                    let t = self.expm1(&abs);
                    let one_plus = self.add(&t, &float(1.0));
                    let ratio = self.div(&t, &one_plus);
                    let sum = self.add(&t, &ratio);
                    self.mul(&sum, &float(0.5))
                } else {
                    let e = self.exp(&abs);
                    let inv = self.div(&float(1.0), &e);
                    let sum = self.add(&e, &inv);
                    self.mul(&sum, &float(0.5))
                };
                // Delay overflow until the true result exceeds f32::MAX.
                let adjusted = self.sub(&abs, &float(std::f32::consts::LN_2));
                let large = self.exp(&adjusted);
                let big = self.pred("gt", &abs, &float(80.0));
                let value = self.select(&big, &large, &normal);
                if op == Sinh {
                    self.sign(&value, x)
                } else {
                    value
                }
            }
            Tanh => {
                let abs = self.abs(x);
                let twice = self.mul(&abs, &float(-2.0));
                let t = self.expm1(&twice);
                let numerator = self.f("neg.f32", &[&t]);
                let denominator = self.add(&t, &float(2.0));
                let value = self.div(&numerator, &denominator);
                self.sign(&value, x)
            }
            Asinh => {
                let a = self.abs(x);
                let square = self.mul(&a, &a);
                let radicand = self.add(&square, &float(1.0));
                let root = self.f("sqrt.rn.f32", &[&radicand]);
                let denominator = self.add(&root, &float(1.0));
                let fraction = self.div(&square, &denominator);
                let argument = self.add(&a, &fraction);
                let normal = self.log1p(&argument);
                let log = self.log(&a);
                let large = self.add(&log, &float(std::f32::consts::LN_2));
                let big = self.pred("gt", &a, &float(1.0e18));
                let result = self.select(&big, &large, &normal);
                self.sign(&result, x)
            }
            Acosh => {
                let lower = self.sub(x, &float(1.0));
                let upper = self.add(x, &float(1.0));
                let square = self.mul(&lower, &upper);
                let root = self.f("sqrt.rn.f32", &[&square]);
                let argument = self.add(&lower, &root);
                let normal = self.log1p(&argument);
                let log = self.log(x);
                let large = self.add(&log, &float(std::f32::consts::LN_2));
                let big = self.pred("gt", x, &float(1.0e18));
                let result = self.select(&big, &large, &normal);
                let invalid = self.pred("lt", x, &float(1.0));
                self.select(&invalid, &float(f32::NAN), &result)
            }
            Atanh => {
                let a = self.abs(x);
                let numerator = self.mul(&a, &float(2.0));
                let denominator = self.sub(&float(1.0), &a);
                let fraction = self.div(&numerator, &denominator);
                let log = self.log1p(&fraction);
                let result = self.mul(&log, &float(0.5));
                self.sign(&result, x)
            }
        }
    }
    pub(super) fn pow(&mut self, x: &str, y: &str) -> String {
        let abs = self.abs(x);
        let ay = self.abs(y);
        let log = self.log(&abs);
        let exponent = self.mul(y, &log);
        let magnitude = self.exp(&exponent);
        let truncated = self.f("cvt.rzi.f32.f32", &[y]);
        let integer = self.pred("eq", y, &truncated);
        let small = self.pred("lt", &ay, &float(16777216.0));
        let bits = self.calc("cvt.rzi.u32.f32", "u32", &[&ay]);
        let bit = self.calc("and.b32", "b32", &[&bits, "1"]);
        let odd = self.calc("setp.ne.u32", "pred", &[&bit, "0"]);
        let odd = self.calc("and.pred", "pred", &[&small, &odd]);
        let negative = self.negative(x);
        let signed = self.calc("and.pred", "pred", &[&negative, &odd]);
        let negated = self.f("neg.f32", &[&magnitude]);
        let mut result = self.select(&signed, &negated, &magnitude);
        let fractional = self.calc("not.pred", "pred", &[&integer]);
        let negative_finite = self.pred("lt", x, &float(0.0));
        let invalid = self.calc("and.pred", "pred", &[&negative_finite, &fractional]);
        result = self.select(&invalid, &float(f32::NAN), &result);
        // Infinite exponents depend solely on |x| relative to one.
        let y_inf = self.pred("eq", &ay, &float(f32::INFINITY));
        let greater = self.pred("gt", &abs, &float(1.0));
        let negative_y = self.pred("lt", y, &float(0.0));
        let grow = self.calc("xor.pred", "pred", &[&greater, &negative_y]);
        let infinite_exponent = self.select(&grow, &float(f32::INFINITY), &float(0.0));
        result = self.select(&y_inf, &infinite_exponent, &result);
        // ±0 and ±Inf bases: sign only survives odd integral exponents.
        let zero = self.pred("eq", &abs, &float(0.0));
        let infinity = self.pred("eq", &abs, &float(f32::INFINITY));
        let special = self.calc("or.pred", "pred", &[&zero, &infinity]);
        let grows = self.calc("xor.pred", "pred", &[&infinity, &negative_y]);
        let value = self.select(&grows, &float(f32::INFINITY), &float(0.0));
        let negated = self.f("neg.f32", &[&value]);
        let value = self.select(&signed, &negated, &value);
        result = self.select(&special, &value, &result);
        let nan = self.pred("nan", x, y);
        result = self.select(&nan, &float(f32::NAN), &result);
        let abs_one = self.pred("eq", &abs, &float(1.0));
        let one_inf = self.calc("and.pred", "pred", &[&abs_one, &y_inf]);
        result = self.select(&one_inf, &float(1.0), &result);
        let x_one = self.pred("eq", x, &float(1.0));
        let y_zero = self.pred("eq", y, &float(0.0));
        let identity = self.calc("or.pred", "pred", &[&x_one, &y_zero]);
        self.select(&identity, &float(1.0), &result)
    }
    pub(super) fn remainder(&mut self, a: &str, b: &str) -> String {
        // Binary long division avoids overflowing a/b or losing the fractional
        // quotient when its magnitude exceeds f32's integer precision.
        let x = self.abs(a);
        let original = self.abs(b);
        let divisor = self.f("mov.f32", &[&original]);
        let end = self.label();
        let up = self.label();
        let down = self.label();
        let finish = self.label();
        let result = self.f("mov.f32", &[&float(f32::NAN)]);
        let invalid = self.pred("nan", a, b);
        writeln!(self.out, "    @{invalid} bra {end};").unwrap();
        let invalid = self.pred("eq", &x, &float(f32::INFINITY));
        writeln!(self.out, "    @{invalid} bra {end};").unwrap();
        let invalid = self.pred("eq", &original, &float(0.0));
        writeln!(self.out, "    @{invalid} bra {end};").unwrap();
        let small = self.pred("lt", &x, &original);
        writeln!(self.out, "    @{small} bra {finish};\n{up}:").unwrap();
        let twice = self.mul(&divisor, &float(2.0));
        let bigger = self.pred("gt", &twice, &x);
        writeln!(self.out, "    @{bigger} bra {down};").unwrap();
        self.emit("mov.f32", &divisor, &[&twice]);
        writeln!(self.out, "    bra {up};\n{down}:").unwrap();
        let subtract = self.pred("ge", &x, &divisor);
        let difference = self.sub(&x, &divisor);
        self.emit("selp.f32", &x, &[&difference, &x, &subtract]);
        let last = self.pred("eq", &divisor, &original);
        writeln!(self.out, "    @{last} bra {finish};").unwrap();
        self.emit("mul.rn.f32", &divisor, &[&divisor, &float(0.5)]);
        writeln!(self.out, "    bra {down};\n{finish}:").unwrap();
        let signed = self.sign(&x, a);
        self.emit("mov.f32", &result, &[&signed]);
        writeln!(self.out, "{end}:").unwrap();
        result
    }
}
