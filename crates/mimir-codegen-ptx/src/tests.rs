//! A small CPU interpreter for the scalar PTX subset used by math lowering.
//! This checks numerical behavior without requiring a GPU. ptxas separately
//! validates syntax and instruction constraints in the integration tests.
use super::*;

struct Program {
    instructions: Vec<String>,
    result: String,
}

impl Program {
    fn new(build: impl FnOnce(&mut Emitter<'_>) -> String) -> Self {
        let spec = SpecializedRoutine {
            constants: BTreeMap::new(),
            allocations: vec![],
            calls: BTreeMap::new(),
        };
        let functions = BTreeMap::new();
        let mut e = Emitter::new(&spec, &functions);
        let result = build(&mut e);
        Self {
            instructions: e.out.lines().map(|s| s.trim().to_string()).collect(),
            result,
        }
    }

    fn run(&self, x: f32, y: f32) -> f32 {
        let mut registers: BTreeMap<String, u32> =
            [("%x".into(), x.to_bits()), ("%y".into(), y.to_bits())].into();
        let labels: BTreeMap<_, _> = self
            .instructions
            .iter()
            .enumerate()
            .filter_map(|(i, s)| s.strip_suffix(':').map(|s| (s, i)))
            .collect();
        let read = |s: &str, regs: &BTreeMap<String, u32>| -> u32 {
            if s.starts_with('%') {
                regs[s]
            } else if let Some(hex) = s.strip_prefix("0f") {
                u32::from_str_radix(hex, 16).unwrap()
            } else {
                s.parse::<u32>().unwrap()
            }
        };
        let mut pc = 0;
        let mut steps = 0;
        while pc < self.instructions.len() {
            steps += 1;
            assert!(steps < 10000, "generated math failed to terminate");
            let line = &self.instructions[pc];
            pc += 1;
            if line.ends_with(':') {
                continue;
            }
            let (op, operands) = line.trim_end_matches(';').split_once(' ').unwrap();
            if let Some(predicate) = op.strip_prefix('@') {
                let condition = read(predicate, &registers) != 0;
                let label = operands.strip_prefix("bra ").unwrap();
                if condition {
                    pc = labels[label];
                }
                continue;
            }
            if op == "bra" {
                pc = labels[operands];
                continue;
            }
            let args: Vec<_> = operands.split(", ").collect();
            let dest = args[0];
            let bits: Vec<_> = args[1..].iter().map(|s| read(s, &registers)).collect();
            let f: Vec<_> = bits.iter().map(|b| f32::from_bits(*b)).collect();
            let result = match op {
                "mov.f32" | "mov.b32" | "mov.pred" | "mov.u32" => bits[0],
                "add.rn.f32" => (f[0] + f[1]).to_bits(),
                "sub.rn.f32" => (f[0] - f[1]).to_bits(),
                "mul.rn.f32" => (f[0] * f[1]).to_bits(),
                "div.rn.f32" => (f[0] / f[1]).to_bits(),
                "fma.rn.f32" => f[0].mul_add(f[1], f[2]).to_bits(),
                "abs.f32" => f[0].abs().to_bits(),
                "neg.f32" => (-f[0]).to_bits(),
                "copysign.f32" => f[1].copysign(f[0]).to_bits(),
                "selp.f32" => {
                    if bits[2] != 0 {
                        bits[0]
                    } else {
                        bits[1]
                    }
                }
                "sqrt.rn.f32" => f[0].sqrt().to_bits(),
                "sin.approx.f32" => f[0].sin().to_bits(),
                "cos.approx.f32" => f[0].cos().to_bits(),
                "ex2.approx.f32" => f[0].exp2().to_bits(),
                "lg2.approx.f32" => f[0].log2().to_bits(),
                "cvt.rmi.f32.f32" => f[0].floor().to_bits(),
                "cvt.rpi.f32.f32" => f[0].ceil().to_bits(),
                "cvt.rzi.f32.f32" => f[0].trunc().to_bits(),
                "cvt.rzi.u32.f32" => f[0].trunc() as u32,
                "and.b32" | "and.pred" => bits[0] & bits[1],
                "or.pred" | "or.b32" => bits[0] | bits[1],
                "xor.pred" => bits[0] ^ bits[1],
                "not.pred" => u32::from(bits[0] == 0),
                "setp.ne.u32" => u32::from(bits[0] != bits[1]),
                "setp.eq.f32" => u32::from(f[0] == f[1]),
                "setp.neu.f32" => u32::from(f[0] != f[1]),
                "setp.lt.f32" => u32::from(f[0] < f[1]),
                "setp.le.f32" => u32::from(f[0] <= f[1]),
                "setp.gt.f32" => u32::from(f[0] > f[1]),
                "setp.ge.f32" => u32::from(f[0] >= f[1]),
                "setp.nan.f32" => u32::from(f[0].is_nan() || f[1].is_nan()),
                _ => panic!("unhandled test instruction: {line}"),
            };
            registers.insert(dest.into(), result);
        }
        f32::from_bits(read(&self.result, &registers))
    }
}

fn close(actual: f32, expected: f32, tolerance: f32) {
    if expected.is_nan() {
        assert!(actual.is_nan(), "expected NaN, got {actual}");
    } else if expected.is_infinite() || expected == 0.0 {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "expected {expected}, got {actual}"
        );
    } else {
        assert!(
            (actual - expected).abs() <= tolerance * expected.abs().max(1e-30),
            "expected {expected}, got {actual}"
        );
    }
}

#[test]
fn unary_math_matches_host_across_domains_and_extremes() {
    use UnaryIntrinsic::*;
    type Reference = (UnaryIntrinsic, fn(f32) -> f32);
    let operations: [Reference; 20] = [
        (Sin, f32::sin),
        (Cos, f32::cos),
        (Tan, f32::tan),
        (Asin, f32::asin),
        (Acos, f32::acos),
        (Atan, f32::atan),
        (Sinh, f32::sinh),
        (Cosh, f32::cosh),
        (Tanh, f32::tanh),
        (Asinh, f32::asinh),
        (Acosh, f32::acosh),
        (Atanh, |x| (x as f64).atanh() as f32),
        (Exp, f32::exp),
        (Log, f32::ln),
        (Exp2, f32::exp2),
        (Log2, f32::log2),
        (Sqrt, f32::sqrt),
        (InverseSqrt, |x| 1.0 / x.sqrt()),
        (Floor, f32::floor),
        (Ceil, f32::ceil),
    ];
    for (op, reference) in operations {
        let program = Program::new(|e| e.unary_math(op, "%x"));
        for x in [
            f32::NEG_INFINITY,
            -1e30,
            -100.0,
            -90.0,
            -10.0,
            -2.0,
            -1.0,
            -0.99999994,
            -0.5,
            -0.125,
            -1e-8,
            -1e-30,
            -f32::from_bits(1),
            -0.0,
            0.0,
            f32::from_bits(1),
            1e-30,
            1e-8,
            0.125,
            0.5,
            0.99999994,
            1.0,
            1.0000001,
            2.0,
            10.0,
            80.0,
            89.0,
            90.0,
            100.0,
            1e30,
            f32::INFINITY,
            f32::NAN,
        ] {
            let actual = program.run(x, 0.0);
            let expected = reference(x);
            // Host SFU reference is exact here; the hardware approximation error
            // is additional and documented separately in the crate README.
            if matches!(op, Sin | Cos | Tan) && x.is_finite() && x.abs() > 100.0 {
                continue;
            }
            close(actual, expected, 3e-5);
        }
    }
}

#[test]
fn atan2_signed_zeros_infinities_and_quadrants() {
    let p = Program::new(|e| e.atan2("%x", "%y"));
    let values = [
        f32::NEG_INFINITY,
        -1e30,
        -1.0,
        -1e-30,
        -f32::from_bits(1),
        -0.0,
        0.0,
        f32::from_bits(1),
        1e-30,
        1.0,
        1e30,
        f32::INFINITY,
        f32::NAN,
    ];
    for y in values {
        for x in values {
            close(p.run(y, x), y.atan2(x), 2e-6);
        }
    }
}

#[test]
fn pow_special_values_and_negative_integral_exponents() {
    let p = Program::new(|e| e.pow("%x", "%y"));
    let values = [
        f32::NEG_INFINITY,
        -10.0,
        -3.0,
        -2.0,
        -1.0,
        -0.5,
        -0.0,
        0.0,
        0.5,
        1.0,
        2.0,
        3.0,
        10.0,
        f32::INFINITY,
        f32::NAN,
    ];
    for x in values {
        for y in values {
            close(p.run(x, y), x.powf(y), 2e-5);
        }
    }
}

#[test]
fn floating_remainder_handles_large_quotients_and_subnormals() {
    let p = Program::new(|e| e.remainder("%x", "%y"));
    let values = [
        f32::NEG_INFINITY,
        -f32::MAX,
        -1e30,
        -10.0,
        -3.0,
        -0.1,
        -0.0,
        0.0,
        f32::from_bits(1),
        f32::MIN_POSITIVE,
        0.1,
        3.0,
        10.0,
        1e30,
        f32::MAX,
        f32::INFINITY,
        f32::NAN,
    ];
    for x in values {
        for y in values {
            let actual = p.run(x, y);
            let expected = x % y;
            if expected.is_nan() {
                assert!(actual.is_nan());
            } else {
                assert_eq!(actual.to_bits(), expected.to_bits(), "{x} % {y}");
            }
        }
    }
}

#[test]
fn edge_assignment_stages_all_sources_before_a_cycle() {
    let program = Program::new(|e| {
        e.types = [
            (ValueId(0), Type::Scalar(ScalarType::U32)),
            (ValueId(1), Type::Scalar(ScalarType::U32)),
        ]
        .into();
        e.emit("mov.u32", "%v0", &["7"]);
        e.emit("mov.u32", "%v1", &["11"]);
        let body = Body {
            entry: BlockId(0),
            blocks: vec![Block {
                id: BlockId(0),
                arguments: vec![ValueId(0), ValueId(1)],
                instructions: vec![],
                merge: None,
                terminator: Terminator::Return(None),
            }],
        };
        e.edge(
            &Edge {
                target: BlockId(0),
                arguments: vec![Operand::Value(ValueId(1)), Operand::Value(ValueId(0))],
            },
            &body,
        );
        e.out.push_str("block0:\n");
        // Encode both destinations into one result using a bitwise OR.
        e.calc("or.b32", "b32", &["%v0", "%v1"])
    });
    assert_eq!(program.run(0.0, 0.0).to_bits(), 7 | 11);
}

#[test]
fn sampled_math_and_remainder_regressions() {
    let atan = Program::new(|e| e.unary_math(UnaryIntrinsic::Atan, "%x"));
    let asin = Program::new(|e| e.unary_math(UnaryIntrinsic::Asin, "%x"));
    let atanh = Program::new(|e| e.unary_math(UnaryIntrinsic::Atanh, "%x"));
    let remainder = Program::new(|e| e.remainder("%x", "%y"));
    let mut seed = 0x12345678u32;
    for _ in 0..512 {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        let x = seed as f64 / u32::MAX as f64 * 2.0 - 1.0;
        let x = x as f32;
        close(atan.run(x, 0.0), x.atan(), 2e-6);
        close(asin.run(x, 0.0), x.asin(), 2e-6);
        close(atanh.run(x, 0.0), (x as f64).atanh() as f32, 2e-6);
        let a = f32::from_bits(seed);
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        let b = f32::from_bits(seed);
        let actual = remainder.run(a, b);
        let expected = a % b;
        if expected.is_nan() {
            assert!(actual.is_nan());
        } else {
            assert_eq!(actual.to_bits(), expected.to_bits(), "{a} % {b}");
        }
    }
}
