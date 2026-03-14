pub mod mat2x2;
pub mod mat3x3;
pub mod mat4x4;

pub trait MatOperations {
    fn identity() -> Self;
    fn zero() -> Self;
    fn det(self) -> f64;
    fn trace(self) -> f64;
    fn transpose(self) -> Self;
    fn inverse(self) -> Option<Self>
    where
        Self: Sized;
    fn is_invertible(self) -> bool;
    fn adjugate(self) -> Self;
    fn lerp(a: Self, b: Self, t: f64) -> Self;
    fn norm(self) -> f64;
    fn normalize(self) -> Self;
}
