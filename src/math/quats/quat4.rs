use crate::math::quats::QuatOperations;
use crate::math::vecs::VecOperations;
use crate::math::vecs::vec3::Vec3;
use std::fmt;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// A four-dimensional quaternion {x, y, z, w}. Used for 3D rotations and orientation representation.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Default)]
pub struct Quat4 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub w: f64,
}

impl Quat4 {
    /// Creating new quaternion [Quat4].
    ///
    /// # Examples
    /// ```
    /// use engine::math::quats::quat4::Quat4;
    ///
    /// let q = Quat4::new(1.0, 2.0, 3.0, 4.0);
    /// assert_eq!(q.x, 1.0);
    /// assert_eq!(q.y, 2.0);
    /// assert_eq!(q.z, 3.0);
    /// assert_eq!(q.w, 4.0);
    /// ```
    pub fn new(x: f64, y: f64, z: f64, w: f64) -> Self {
        Quat4 { x, y, z, w }
    }
}

impl fmt::Display for Quat4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let x = format!("{:.10}", self.x)
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_string();
        let y = format!("{:.10}", self.y)
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_string();
        let z = format!("{:.10}", self.z)
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_string();
        let w = format!("{:.10}", self.w)
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_string();
        write!(f, "Quat4 ({}, {}, {}, {})", x, y, z, w)
    }
}

// --- Arithmetic Implementation ---

impl Add for Quat4 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
            w: self.w + rhs.w,
        }
    }
}

impl Sub for Quat4 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
            w: self.w - rhs.w,
        }
    }
}

/// Hamilton product for [Quat4].
impl Mul for Quat4 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            x: self.w * rhs.x + self.x * rhs.w + self.y * rhs.z - self.z * rhs.y,
            y: self.w * rhs.y - self.x * rhs.z + self.y * rhs.w + self.z * rhs.x,
            z: self.w * rhs.z + self.x * rhs.y - self.y * rhs.x + self.z * rhs.w,
            w: self.w * rhs.w - self.x * rhs.x - self.y * rhs.y - self.z * rhs.z,
        }
    }
}

impl Mul<f64> for Quat4 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: f64) -> Self::Output {
        Self {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
            w: self.w * rhs,
        }
    }
}

impl Div<f64> for Quat4 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: f64) -> Self::Output {
        Self {
            x: self.x / rhs,
            y: self.y / rhs,
            z: self.z / rhs,
            w: self.w / rhs,
        }
    }
}

impl Neg for Quat4 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self::Output {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
            w: -self.w,
        }
    }
}

impl AddAssign for Quat4 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
        self.w += rhs.w;
    }
}

impl SubAssign for Quat4 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.x -= rhs.x;
        self.y -= rhs.y;
        self.z -= rhs.z;
        self.w -= rhs.w;
    }
}

impl MulAssign for Quat4 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl MulAssign<f64> for Quat4 {
    #[inline]
    fn mul_assign(&mut self, rhs: f64) {
        self.x *= rhs;
        self.y *= rhs;
        self.z *= rhs;
        self.w *= rhs;
    }
}

impl DivAssign<f64> for Quat4 {
    #[inline]
    fn div_assign(&mut self, rhs: f64) {
        self.x /= rhs;
        self.y /= rhs;
        self.z /= rhs;
        self.w /= rhs;
    }
}

// --- QuatOperations Implementation ---

impl QuatOperations for Quat4 {
    /// Calculates the squared norm of [Quat4].
    ///
    /// # Examples
    /// ```
    /// # use engine::math::quats::{quat4::Quat4, QuatOperations};
    /// let q = Quat4::new(1.0, 1.0, 1.0, 1.0);
    /// assert_eq!(q.norm_sq(), 4.0);
    /// ```
    #[inline]
    fn norm_sq(self) -> f64 {
        self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w
    }

    /// Calculates the length of [Quat4].
    ///
    /// # Examples
    /// ```
    /// # use engine::math::quats::{quat4::Quat4, QuatOperations};
    /// let q = Quat4::new(2.0, 0.0, 0.0, 0.0);
    /// assert_eq!(q.length(), 2.0);
    /// ```
    #[inline]
    fn length(self) -> f64 {
        self.norm_sq().sqrt()
    }

    /// Normalize [Quat4]. Resulting length will be 1.0.
    ///
    /// # Examples
    /// ```
    /// # use engine::math::quats::{quat4::Quat4, QuatOperations};
    /// let q = Quat4::new(2.0, 0.0, 0.0, 0.0).norm();
    /// assert!((q.length() - 1.0).abs() < f64::EPSILON);
    /// ```
    #[inline]
    fn norm(self) -> Self {
        let len = self.length();
        if len > f64::EPSILON {
            self / len
        } else {
            Self::zero()
        }
    }

    /// Returns the conjugate of [Quat4].
    ///
    /// # Examples
    /// ```
    /// # use engine::math::quats::{quat4::Quat4, QuatOperations};
    /// let q = Quat4::new(1.0, 2.0, 3.0, 4.0);
    /// assert_eq!(q.conj(), Quat4::new(-1.0, -2.0, -3.0, 4.0));
    /// ```
    #[inline]
    fn conj(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
            w: self.w,
        }
    }

    /// Returns the inverse of [Quat4].
    ///
    /// # Examples
    /// ```
    /// # use engine::math::quats::{quat4::Quat4, QuatOperations};
    /// let q = Quat4::new(1.0, 0.0, 0.0, 0.0);
    /// assert!(q.inverse().is_some());
    /// ```
    #[inline]
    fn inverse(self) -> Option<Self> {
        let n2 = self.norm_sq();
        if n2 > f64::EPSILON {
            Some(self.conj() / n2)
        } else {
            None
        }
    }

    /// Calculates the dot product of two [Quat4].
    ///
    /// # Examples
    /// ```
    /// # use engine::math::quats::{quat4::Quat4, QuatOperations};
    /// let a = Quat4::new(1.0, 2.0, 3.0, 4.0);
    /// let b = Quat4::new(1.0, 2.0, 3.0, 4.0);
    /// assert_eq!(a.dot(b), 30.0);
    /// ```
    #[inline]
    fn dot(self, other: Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w
    }

    /// Returns the identity rotation quaternion.
    ///
    /// # Examples
    /// ```
    /// # use engine::math::quats::{quat4::Quat4, QuatOperations};
    /// let q = Quat4::identity();
    /// assert_eq!(q, Quat4::new(0.0, 0.0, 0.0, 1.0));
    /// ```
    #[inline]
    fn identity() -> Self {
        Self::new(0.0, 0.0, 0.0, 1.0)
    }

    /// Returns a zero quaternion.
    ///
    /// # Examples
    /// ```
    /// # use engine::math::quats::{quat4::Quat4, QuatOperations};
    /// let q = Quat4::zero();
    /// assert_eq!(q, Quat4::new(0.0, 0.0, 0.0, 0.0));
    /// ```
    #[inline]
    fn zero() -> Self {
        Self::new(0.0, 0.0, 0.0, 0.0)
    }

    /// Creates a [Quat4] from a rotation axis and angle in radians.
    ///
    /// # Examples
    /// ```
    /// # use engine::math::quats::{quat4::Quat4, QuatOperations};
    /// # use engine::math::vecs::vec3::Vec3;
    /// let axis = Vec3::new(1.0, 0.0, 0.0);
    /// let q = Quat4::from_axis_angle(axis, std::f64::consts::PI);
    /// ```
    fn from_axis_angle(axis: Vec3, rads: f64) -> Self {
        let (s, c) = (rads * 0.5).sin_cos();
        let n_axis = axis.norm();
        Self::new(n_axis.x * s, n_axis.y * s, n_axis.z * s, c)
    }

    /// Linear interpolation between two [Quat4] by a factor `t`.
    ///
    /// # Examples
    /// ```
    /// # use engine::math::quats::{quat4::Quat4, QuatOperations};
    /// let a = Quat4::identity();
    /// let b = Quat4::new(1.0, 0.0, 0.0, 0.0);
    /// let res = a.lerp(b, 0.5);
    /// ```
    #[inline]
    fn lerp(self, other: Self, t: f64) -> Self {
        (self * (1.0 - t) + other * t).norm()
    }

    /// Spherical linear interpolation between two [Quat4] by a factor `t`.
    ///
    /// # Examples
    /// ```
    /// # use engine::math::quats::{quat4::Quat4, QuatOperations};
    /// let a = Quat4::identity();
    /// let b = Quat4::identity();
    /// let res = a.slerp(b, 0.5);
    /// ```
    fn slerp(self, mut other: Self, t: f64) -> Self {
        let mut dot = self.dot(other);
        if dot < 0.0 {
            other = -other;
            dot = -dot;
        }
        if dot > 0.9995 {
            return self.lerp(other, t);
        }
        let theta_0 = dot.acos();
        let theta = theta_0 * t;
        let s0 = (theta_0 - theta).sin() / theta_0.sin();
        let s1 = theta.sin() / theta_0.sin();
        (self * s0) + (other * s1)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::math::vecs::vec3::Vec3;

    #[test]
    fn test_quat4_display() {
        let a = Quat4::new(2.0, 3.0, 4.0, 5.0);
        assert_eq!(format!("{}", a), "Quat4 (2, 3, 4, 5)");

        let b = Quat4::new(1.5, 0.25, 0.125, 1.0);
        assert_eq!(format!("{}", b), "Quat4 (1.5, 0.25, 0.125, 1)");

        let c = Quat4::new(1.1234567890123, 0.0, 0.0, 0.0000000001);
        assert_eq!(format!("{}", c), "Quat4 (1.123456789, 0, 0, 0.0000000001)");
    }

    #[test]
    fn test_quat4_add() {
        let a = Quat4::new(1.0, 1.0, 1.0, 1.0);
        let b = Quat4::new(2.0, 2.0, 2.0, 2.0);
        assert_eq!(a + b, Quat4::new(3.0, 3.0, 3.0, 3.0));
    }

    #[test]
    fn test_quat4_sub() {
        let a = Quat4::new(5.0, 5.0, 5.0, 5.0);
        let b = Quat4::new(2.0, 2.0, 2.0, 2.0);
        assert_eq!(a - b, Quat4::new(3.0, 3.0, 3.0, 3.0));
    }

    #[test]
    fn test_quat4_mul_hamilton() {
        let i = Quat4::new(1.0, 0.0, 0.0, 0.0);
        let j = Quat4::new(0.0, 1.0, 0.0, 0.0);
        let k = Quat4::new(0.0, 0.0, 1.0, 0.0);
        assert_eq!(i * j, k);
        assert_eq!(i * i, Quat4::new(0.0, 0.0, 0.0, -1.0));
    }

    #[test]
    fn test_quat4_mul_scalar() {
        let a = Quat4::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(a * 2.0, Quat4::new(2.0, 4.0, 6.0, 8.0));
    }

    #[test]
    fn test_quat4_div_scalar() {
        let a = Quat4::new(2.0, 4.0, 6.0, 8.0);
        assert_eq!(a / 2.0, Quat4::new(1.0, 2.0, 3.0, 4.0));
    }

    #[test]
    fn test_quat4_neg() {
        let a = Quat4::new(1.0, -2.0, 3.0, -4.0);
        assert_eq!(-a, Quat4::new(-1.0, 2.0, -3.0, 4.0));
    }

    #[test]
    fn test_quat4_add_assign() {
        let mut a = Quat4::new(1.0, 1.0, 1.0, 1.0);
        a += Quat4::new(2.0, 2.0, 2.0, 2.0);
        assert_eq!(a, Quat4::new(3.0, 3.0, 3.0, 3.0));
    }

    #[test]
    fn test_quat4_sub_assign() {
        let mut a = Quat4::new(5.0, 5.0, 5.0, 5.0);
        a -= Quat4::new(2.0, 2.0, 2.0, 2.0);
        assert_eq!(a, Quat4::new(3.0, 3.0, 3.0, 3.0));
    }

    #[test]
    fn test_quat4_mul_assign() {
        let mut i = Quat4::new(1.0, 0.0, 0.0, 0.0);
        let j = Quat4::new(0.0, 1.0, 0.0, 0.0);
        i *= j;
        assert_eq!(i, Quat4::new(0.0, 0.0, 1.0, 0.0));
    }

    #[test]
    fn test_quat4_mul_assign_scalar() {
        let mut a = Quat4::new(1.0, 2.0, 3.0, 4.0);
        a *= 2.0;
        assert_eq!(a, Quat4::new(2.0, 4.0, 6.0, 8.0));
    }

    #[test]
    fn test_quat4_div_assign_scalar() {
        let mut a = Quat4::new(2.0, 4.0, 6.0, 8.0);
        a /= 2.0;
        assert_eq!(a, Quat4::new(1.0, 2.0, 3.0, 4.0));
    }

    #[test]
    fn test_quat4_norm_sq() {
        let q = Quat4::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(q.norm_sq(), 30.0);
    }

    #[test]
    fn test_quat4_length() {
        let q = Quat4::new(0.0, 3.0, 4.0, 0.0);
        assert_eq!(q.length(), 5.0);
    }

    #[test]
    fn test_quat4_norm() {
        let q = Quat4::new(2.0, 2.0, 2.0, 2.0).norm();
        assert!((q.length() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_quat4_conj() {
        let q = Quat4::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(q.conj(), Quat4::new(-1.0, -2.0, -3.0, 4.0));
    }

    #[test]
    fn test_quat4_inverse() {
        let q = Quat4::new(1.0, 2.0, 3.0, 4.0);
        let inv = q.inverse().expect("Inverse should exist");
        let res = q * inv;
        assert!((res.x).abs() < 1e-15);
        assert!((res.y).abs() < 1e-15);
        assert!((res.z).abs() < 1e-15);
        assert!((res.w - 1.0).abs() < 1e-15);
    }

    #[test]
    fn test_quat4_dot() {
        let a = Quat4::new(1.0, 2.0, 3.0, 4.0);
        let b = Quat4::new(0.5, 0.5, 0.5, 0.5);
        assert_eq!(a.dot(b), 5.0);
    }

    #[test]
    fn test_quat4_identity() {
        let q = Quat4::identity();
        assert_eq!(q, Quat4::new(0.0, 0.0, 0.0, 1.0));
    }

    #[test]
    fn test_quat4_zero() {
        let q = Quat4::zero();
        assert_eq!(q, Quat4::new(0.0, 0.0, 0.0, 0.0));
    }

    #[test]
    fn test_quat4_from_axis_angle() {
        let axis = Vec3::new(1.0, 0.0, 0.0);
        let q = Quat4::from_axis_angle(axis, std::f64::consts::PI); // 180 deg
        assert!((q.x - 1.0).abs() < 1e-15);
        assert!(q.w.abs() < 1e-15);
    }

    #[test]
    fn test_quat4_lerp() {
        let a = Quat4::new(0.0, 0.0, 0.0, 1.0);
        let b = Quat4::new(1.0, 0.0, 0.0, 0.0);
        let res = a.lerp(b, 0.5);
        assert!((res.x - res.w).abs() < f64::EPSILON);
    }

    #[test]
    fn test_quat4_slerp() {
        let a = Quat4::identity();
        let b = Quat4::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), std::f64::consts::PI / 2.0);
        let res = a.slerp(b, 0.5);
        let expected = Quat4::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), std::f64::consts::PI / 4.0);
        assert!((res.dot(expected).abs() - 1.0).abs() < 1e-10);
    }
}
