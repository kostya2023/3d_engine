use crate::math::vecs::VecOperations;
use std::ops::{Add, Sub, Mul, Div, Neg, AddAssign, SubAssign, MulAssign, DivAssign};
use std::fmt;

/// A four-dimensional vector. Useful for homogeneous coordinates and 4D math.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Default)]
pub struct Vec4 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub w: f64,
}

impl Vec4 {
    /// Creating new vector [Vec4].
    ///
    /// # Examples
    /// ```
    /// use engine::math::vecs::vec4::Vec4;
    ///
    /// let v = Vec4::new(1.0, 2.0, 3.0, 4.0);
    /// assert_eq!(v.x, 1.0);
    /// assert_eq!(v.y, 2.0);
    /// assert_eq!(v.z, 3.0);
    /// assert_eq!(v.w, 4.0);
    /// ```
    pub fn new(x: f64, y: f64, z: f64, w: f64) -> Self {
        Vec4 { x, y, z, w }
    }
}

impl fmt::Display for Vec4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let x = format!("{:.10}", self.x).trim_end_matches('0').trim_end_matches('.').to_string();
        let y = format!("{:.10}", self.y).trim_end_matches('0').trim_end_matches('.').to_string();
        let z = format!("{:.10}", self.z).trim_end_matches('0').trim_end_matches('.').to_string();
        let w = format!("{:.10}", self.w).trim_end_matches('0').trim_end_matches('.').to_string();
        write!(f, "Vec4 ({}, {}, {}, {})", x, y, z, w)
    }
}

// --- Arithmetic Implementation ---

impl Add for Vec4 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self { x: self.x + rhs.x, y: self.y + rhs.y, z: self.z + rhs.z, w: self.w + rhs.w }
    }
}

impl Sub for Vec4 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self { x: self.x - rhs.x, y: self.y - rhs.y, z: self.z - rhs.z, w: self.w - rhs.w }
    }
}

impl Mul for Vec4 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self { x: self.x * rhs.x, y: self.y * rhs.y, z: self.z * rhs.z, w: self.w * rhs.w }
    }
}

impl Mul<f64> for Vec4 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: f64) -> Self::Output {
        Self { x: self.x * rhs, y: self.y * rhs, z: self.z * rhs, w: self.w * rhs }
    }
}

impl Div for Vec4 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        Self { x: self.x / rhs.x, y: self.y / rhs.y, z: self.z / rhs.z, w: self.w / rhs.w }
    }
}

impl Div<f64> for Vec4 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: f64) -> Self::Output {
        Self { x: self.x / rhs, y: self.y / rhs, z: self.z / rhs, w: self.w / rhs }
    }
}

impl Neg for Vec4 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self::Output {
        Self { x: -self.x, y: -self.y, z: -self.z, w: -self.w }
    }
}

impl AddAssign for Vec4 {
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
        self.w += rhs.w;
    }
}

impl SubAssign for Vec4 {
    fn sub_assign(&mut self, rhs: Self) {
        self.x -= rhs.x;
        self.y -= rhs.y;
        self.z -= rhs.z;
        self.w -= rhs.w;
    }
}

impl MulAssign for Vec4 {
    fn mul_assign(&mut self, rhs: Self) {
        self.x *= rhs.x;
        self.y *= rhs.y;
        self.z *= rhs.z;
        self.w *= rhs.w;
    }
}

impl MulAssign<f64> for Vec4 {
    #[inline]
    fn mul_assign(&mut self, rhs: f64) {
        self.x *= rhs;
        self.y *= rhs;
        self.z *= rhs;
        self.w *= rhs;
    }
}

impl DivAssign for Vec4 {
    fn div_assign(&mut self, rhs: Self) {
        self.x /= rhs.x;
        self.y /= rhs.y;
        self.z /= rhs.z;
        self.w /= rhs.w;
    }
}

impl DivAssign<f64> for Vec4 {
    #[inline]
    fn div_assign(&mut self, rhs: f64) {
        self.x /= rhs;
        self.y /= rhs;
        self.z /= rhs;
        self.w /= rhs;
    }
}

// --- VecOperations Implementation ---

impl VecOperations for Vec4 {
    /// Calculates the dot product of two [Vec4].
    ///
    /// # Examples
    /// ```
    /// # use engine::math::vecs::vec4::Vec4;
    /// # use engine::math::vecs::VecOperations;
    /// let a = Vec4::new(2.0, 2.0, 2.0, 2.0);
    /// let b = Vec4::new(2.0, 2.0, 2.0, 2.0);
    /// assert_eq!(a.dot(b), 16.0);
    /// ```
    #[inline]
    fn dot(self, other: Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w
    }

    /// Calculates square length of [Vec4]. (Faster than [length])
    ///
    /// # Examples
    /// ```
    /// # use engine::math::vecs::vec4::Vec4;
    /// # use engine::math::vecs::VecOperations;
    /// let a = Vec4::new(2.0, 2.0, 2.0, 2.0);
    /// assert_eq!(a.length2(), 16.0);
    /// ```
    #[inline]
    fn length2(self) -> f64 {
        self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w
    }

    /// Calculates sqrt length of [Vec4].
    ///
    /// # Examples
    /// ```
    /// # use engine::math::vecs::vec4::Vec4;
    /// # use engine::math::vecs::VecOperations;
    /// let a = Vec4::new(2.0, 2.0, 2.0, 2.0);
    /// assert_eq!(a.length(), 16.0f64.sqrt());
    /// ```
    #[inline]
    fn length(self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w).sqrt()
    }

    /// Normalize [Vec4]. Resulting length will be 1.0.
    ///
    /// # Examples
    /// ```
    /// # use engine::math::vecs::vec4::Vec4;
    /// # use engine::math::vecs::VecOperations;
    /// let a = Vec4::new(2.0, 2.0, 2.0, 2.0).norm();
    /// assert!((a.length() - 1.0).abs() < f64::EPSILON);
    /// ```
    #[inline]
    fn norm(self) -> Self {
        self / self.length()
    }

    /// Calculates the Euclidean distance between two [Vec4].
    ///
    /// # Examples
    /// ```
    /// # use engine::math::vecs::vec4::Vec4;
    /// # use engine::math::vecs::VecOperations;
    /// let a = Vec4::new(2.0, 2.0, 2.0, 2.0);
    /// let b = Vec4::new(2.0, 2.0, 2.0, 2.0);
    /// assert_eq!(a.distance(b), 0.0);
    /// ```
    #[inline]
    fn distance(self, other: Self) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2) + (self.z - other.z).powi(2) + (self.w - other.w).powi(2)).sqrt()
    }

    /// Linear interpolation between two [Vec4] by a factor `t`.
    ///
    /// # Examples
    /// ```
    /// # use engine::math::vecs::vec4::Vec4;
    /// # use engine::math::vecs::VecOperations;
    /// let a = Vec4::new(0.0, 0.0, 0.0, 0.0);
    /// let b = Vec4::new(2.0, 2.0, 2.0, 2.0);
    /// assert_eq!(a.lerp(b, 0.5), Vec4::new(1.0, 1.0, 1.0, 1.0));
    /// ```
    #[inline]
    fn lerp(self, other: Self, t: f64) -> Self {
        self * (1.0 - t) + other * t
    }

    /// Returns a [Vec4] containing the minimum components of two vectors.
    ///
    /// # Examples
    /// ```
    /// # use engine::math::vecs::vec4::Vec4;
    /// # use engine::math::vecs::VecOperations;
    /// let a = Vec4::new(0.0, 5.0, -1.0, 0.0);
    /// let b = Vec4::new(1.0, 2.0, 2.0, 1.0);
    /// assert_eq!(a.min(b), Vec4::new(0.0, 2.0, -1.0, 0.0));
    /// ```
    #[inline]
    fn min(self, other: Self) -> Self {
        Self { x: self.x.min(other.x), y: self.y.min(other.y), z: self.z.min(other.z), w: self.w.min(other.w) }
    }

    /// Returns a [Vec4] containing the maximum components of two vectors.
    ///
    /// # Examples
    /// ```
    /// # use engine::math::vecs::vec4::Vec4;
    /// # use engine::math::vecs::VecOperations;
    /// let a = Vec4::new(0.0, 5.0, -1.0, 0.0);
    /// let b = Vec4::new(1.0, 2.0, 2.0, 1.0);
    /// assert_eq!(a.max(b), Vec4::new(1.0, 5.0, 2.0, 1.0));
    /// ```
    #[inline]
    fn max(self, other: Self) -> Self {
        Self { x: self.x.max(other.x), y: self.y.max(other.y), z: self.z.max(other.z), w: self.w.max(other.w) }
    }

    /// Clamps the [Vec4] components between `min` and `max` vectors.
    ///
    /// # Examples
    /// ```
    /// # use engine::math::vecs::vec4::Vec4;
    /// # use engine::math::vecs::VecOperations;
    /// let min = Vec4::new(-1.0, -1.0, -1.0, -1.0);
    /// let max = Vec4::new(1.0, 1.0, 1.0, 1.0);
    /// let v = Vec4::new(2.0, -2.0, 0.0, 0.5);
    /// assert_eq!(v.clamp(min, max), Vec4::new(1.0, -1.0, 0.0, 0.5));
    /// ```
    #[inline]
    fn clamp(self, min: Self, max: Self) -> Self {
        Self { x: self.x.clamp(min.x, max.x), y: self.y.clamp(min.y, max.y), z: self.z.clamp(min.z, max.z), w: self.w.clamp(min.w, max.w) }
    }
}


#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_vec4_display() {
        let a = Vec4 { x: 2.0, y: 3.0, z: 4.0, w: 5.0 };
        assert_eq!(format!("{}", a), "Vec4 (2, 3, 4, 5)");

        let b = Vec4 { x: 1.5, y: 0.25, z: 0.125, w: 0.0625 };
        assert_eq!(format!("{}", b), "Vec4 (1.5, 0.25, 0.125, 0.0625)");

        let c = Vec4 { x: 1.1234567890123, y: 0.0000000001, z: 3.141592653589, w: 2.718281828459 }; 
        assert_eq!(format!("{}", c), "Vec4 (1.123456789, 0.0000000001, 3.1415926536, 2.7182818285)");

        let d = Vec4 { x: 0.0, y: 0.0, z: 0.0, w: 0.0 };
        assert_eq!(format!("{}", d), "Vec4 (0, 0, 0, 0)");
    }

    #[test]
    fn test_vec4_add() {
        let a = Vec4 { x: 2.0, y: 2.0, z: 2.0, w: 2.0 };
        let b = Vec4 { x: 2.0, y: 2.0, z: 2.0, w: 2.0 };
        assert_eq!(a + b, Vec4 { x: 4.0, y: 4.0, z: 4.0, w: 4.0 })
    }

    #[test]
    fn test_vec4_sub() {
        let a = Vec4 { x: 2.0, y: 2.0, z: 2.0, w: 2.0 };
        let b = Vec4 { x: 2.0, y: 2.0, z: 2.0, w: 2.0 };
        assert_eq!(a - b, Vec4 { x: 0.0, y: 0.0, z: 0.0, w: 0.0 })
    }

    #[test]
    fn test_vec4_mul() {
        let a = Vec4 { x: 2.0, y: 2.0, z: 2.0, w: 2.0 };
        let b = Vec4 { x: 2.0, y: 2.0, z: 2.0, w: 2.0 };
        assert_eq!(a * b, Vec4 { x: 4.0, y: 4.0, z: 4.0, w: 4.0 })
    }

    #[test]
    fn test_vec4_mulscalar() {
        let a = Vec4 { x: 2.0, y: 2.0, z: 2.0, w: 2.0 };
        assert_eq!(a * 2.0, Vec4 { x: 4.0, y: 4.0, z: 4.0, w: 4.0 })
    }

    #[test]
    fn test_vec4_div() {
        let a = Vec4 { x: 2.0, y: 2.0, z: 2.0, w: 2.0 };
        let b = Vec4 { x: 2.0, y: 2.0, z: 2.0, w: 2.0 };
        assert_eq!(a / b, Vec4 { x: 1.0, y: 1.0, z: 1.0, w: 1.0 })
    }

    #[test]
    fn test_vec4_divscalar() {
        let a = Vec4 { x: 2.0, y: 2.0, z: 2.0, w: 2.0 };
        assert_eq!(a / 2.0, Vec4 { x: 1.0, y: 1.0, z: 1.0, w: 1.0 })
    }

    #[test]
    fn test_vec4_neg() {
        let a = Vec4 { x: 2.0, y: 2.0, z: 2.0, w: 2.0 };
        assert_eq!(-a, Vec4 { x: -2.0, y: -2.0, z: -2.0, w: -2.0 })
    }

    #[test]
    fn test_vec4_addassign() {
        let mut a = Vec4 { x: 2.0, y: 2.0, z: 2.0, w: 2.0 };
        a += Vec4 { x: 2.0, y: 2.0, z: 2.0, w: 2.0 };
        assert_eq!(a, Vec4 { x: 4.0, y: 4.0, z: 4.0, w: 4.0 })
    }

    #[test]
    fn test_vec4_subassign() {
        let mut a = Vec4 { x: 2.0, y: 2.0, z: 2.0, w: 2.0 };
        a -= Vec4 { x: 2.0, y: 2.0, z: 2.0, w: 2.0 };
        assert_eq!(a, Vec4 { x: 0.0, y: 0.0, z: 0.0, w: 0.0 })
    }

    #[test]
    fn test_vec4_mulassign() {
        let mut a = Vec4 { x: 2.0, y: 2.0, z: 2.0, w: 2.0 };
        a *= Vec4 { x: 2.0, y: 2.0, z: 2.0, w: 2.0 };
        assert_eq!(a, Vec4 { x: 4.0, y: 4.0, z: 4.0, w: 4.0 })
    }

    #[test]
    fn test_vec4_mulassign_scalar() {
        let mut a = Vec4 { x: 2.0, y: 2.0, z: 2.0, w: 2.0 };
        a *= 2.0;
        assert_eq!(a, Vec4 { x: 4.0, y: 4.0, z: 4.0, w: 4.0 })
    }

    #[test]
    fn test_vec4_divassign() {
        let mut a = Vec4 { x: 2.0, y: 2.0, z: 2.0, w: 2.0 };
        a /= Vec4 { x: 2.0, y: 2.0, z: 2.0, w: 2.0 };
        assert_eq!(a, Vec4 { x: 1.0, y: 1.0, z: 1.0, w: 1.0 })
    }

    #[test]
    fn test_vec4_divassign_scalar() {
        let mut a = Vec4 { x: 2.0, y: 2.0, z: 2.0, w: 2.0 };
        a /= 2.0;
        assert_eq!(a, Vec4 { x: 1.0, y: 1.0, z: 1.0, w: 1.0 })
    }

    #[test]
    fn test_vec4_dot() {
        let a = Vec4 { x: 2.0, y: 2.0, z: 2.0, w: 2.0 };
        let b = Vec4 { x: 2.0, y: 2.0, z: 2.0, w: 2.0 };
        assert_eq!(a.dot(b), 16.0)
    }

    #[test]
    fn test_vec4_length2() {
        let a = Vec4 { x: 2.0, y: 2.0, z: 2.0, w: 2.0 };
        assert_eq!(a.length2(), 16.0)
    }

    #[test]
    fn test_vec4_length() {
        let a = Vec4 { x: 2.0, y: 2.0, z: 2.0, w: 2.0 };
        assert_eq!(a.length(), 16.0f64.sqrt())
    }

    #[test]
    fn test_vec4_norm() {
        let a = Vec4 { x: 2.0, y: 2.0, z: 2.0, w: 2.0 }.norm();
        assert!((a.length() - 1.0).abs() < f64::EPSILON)
    }

    #[test]
    fn test_vec4_distance() {
        let a = Vec4 { x: 2.0, y: 2.0, z: 2.0, w: 2.0 };
        let b = Vec4 { x: 2.0, y: 2.0, z: 2.0, w: 2.0 };
        assert_eq!(a.distance(b), 0.0)
    }

    #[test]
    fn test_vec4_lerp() {
        let a = Vec4 { x: 0.0, y: 0.0, z: 0.0, w: 0.0 };
        let b = Vec4 { x: 2.0, y: 2.0, z: 2.0, w: 2.0 };
        assert_eq!(a.lerp(b, 0.5), Vec4 { x: 1.0, y: 1.0, z: 1.0, w: 1.0 })
    }

    #[test]
    fn test_vec4_min() {
        let a = Vec4 { x: 0.0, y: 0.0, z: 0.0, w: 0.0 };
        let b = Vec4 { x: 1.0, y: 1.0, z: 1.0, w: 1.0 };
        assert_eq!(a.min(b), Vec4 { x: 0.0, y: 0.0, z: 0.0, w: 0.0 })
    }

    #[test]
    fn test_vec4_max() {
        let a = Vec4 { x: 0.0, y: 0.0, z: 0.0, w: 0.0 };
        let b = Vec4 { x: 1.0, y: 1.0, z: 1.0, w: 1.0 };
        assert_eq!(a.max(b), Vec4 { x: 1.0, y: 1.0, z: 1.0, w: 1.0 })
    }

    #[test]
    fn test_vec4_clamp() {
        let max = Vec4 { x: 1.0, y: 1.0, z: 1.0, w: 1.0 };
        let min = Vec4 { x: -1.0, y: -1.0, z: -1.0, w: -1.0 };

        let out_min = Vec4 { x: -2.0, y: -2.0, z: -2.0, w: -2.0 };
        let mid = Vec4 { x: 0.0, y: 0.0, z: 0.0, w: 0.0 };
        let out_max = Vec4 { x: 2.0, y: 2.0, z: 2.0, w: 2.0 };

        let mixed_out_min = Vec4 { x: -2.0, y: 0.0, z: 2.0, w: -0.5 };
        let mixed_out_max = Vec4 { x: 2.0, y: 0.0, z: -2.0, w: 0.5 };

        assert_eq!(out_min.clamp(min, max), Vec4 { x: -1.0, y: -1.0, z: -1.0, w: -1.0 });
        assert_eq!(mid.clamp(min, max), Vec4 { x: 0.0, y: 0.0, z: 0.0, w: 0.0 });
        assert_eq!(out_max.clamp(min, max), Vec4 { x: 1.0, y: 1.0, z: 1.0, w: 1.0 });

        assert_eq!(mixed_out_min.clamp(min, max), Vec4 { x: -1.0, y: 0.0, z: 1.0, w: -0.5 });
        assert_eq!(mixed_out_max.clamp(min, max), Vec4 { x: 1.0, y: 0.0, z: -1.0, w: 0.5 });
    }
}
