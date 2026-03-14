pub mod quat4;

use crate::math::vecs::vec3::Vec3;

pub trait QuatOperations {
    fn norm_sq(self) -> f64;
    fn length(self) -> f64;
    fn norm(self) -> Self;
    fn conj(self) -> Self;
    fn inverse(self) -> Option<Self>
    where
        Self: Sized;
    fn dot(self, other: Self) -> f64;
    fn identity() -> Self;
    fn zero() -> Self;
    fn from_axis_angle(axis: Vec3, rads: f64) -> Self;
    fn lerp(self, other: Self, t: f64) -> Self;
    fn slerp(self, other: Self, t: f64) -> Self;
}
