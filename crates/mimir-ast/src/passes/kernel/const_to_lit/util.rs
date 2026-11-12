use std::ops::{Add, Div, Mul, Rem, Sub};

use crate::{ConstExpr, Literal, MimirBinOp, RichConstExpr};

#[inline]
pub fn bin_op_math<
    T: Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T> + Rem<Output = T> + Sized,
>(
    lhs: T,
    op: &MimirBinOp,
    rhs: T,
) -> T {
    match op {
        MimirBinOp::Add => lhs + rhs,
        MimirBinOp::Sub => lhs - rhs,
        MimirBinOp::Mul => lhs * rhs,
        MimirBinOp::Div => lhs / rhs,
        MimirBinOp::Mod => lhs % rhs,
        _ => unreachable!(),
    }
}

pub fn handle_partial_opt_intrinsic<const N: usize>(
    opt_param_arr: [Option<Literal>; N],
    src_params: &mut Box<[RichConstExpr; N]>,
) -> [RichConstExpr; N] {
    opt_param_arr
        .iter()
        .zip(src_params.iter())
        .map(|(opt, const_e)| match opt {
            Some(lit) => RichConstExpr::new(ConstExpr::Literal(*lit), const_e.span.clone())
                .set_p(const_e.is_parenthesized),
            None => const_e.clone(),
        })
        .collect::<Vec<_>>()
        .try_into()
        .unwrap()
}
