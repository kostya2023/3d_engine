pub mod vec2;
pub mod vec3;
pub mod vec4;

pub trait VecOperations {
    fn dot(self, other: Self) -> f64;
    fn length2(self) -> f64;
    fn length(self) -> f64;
    fn norm(self) -> Self;
    fn distance(self, other: Self) -> f64;
    fn lerp(self, other: Self, t: f64) -> Self;
    fn min(self, other: Self) -> Self;
    fn max(self, other: Self) -> Self;
    fn clamp(self, min: Self, max: Self) -> Self;
}
