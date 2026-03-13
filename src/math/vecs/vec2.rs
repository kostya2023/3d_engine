use crate::math::vecs::VecOperations;
use std::ops::{Add, Sub, Mul, Div, Neg, AddAssign, SubAssign, MulAssign, DivAssign};
use std::fmt;

/// A two-dimensional vector. Suitable for points on a plane.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Default)]
pub struct Vec2 {
    pub x: f64,
    pub y: f64,
}

impl Vec2 {
    /// Creating new vector [Vec2].
    /// 
    /// # Examples
    /// ```
    /// use engine::math::vecs::vec2::Vec2;
    /// 
    /// let vector2 = Vec2::new(1.0, 2.0);
    /// assert_eq!(vector2.x, 1.0);
    /// assert_eq!(vector2.y, 2.0);
    /// ```
    pub fn new(x: f64, y: f64) -> Self {
        Vec2 { x, y }
    }
}

impl fmt::Display for Vec2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let x = format!("{:.10}", self.x).trim_end_matches('0').trim_end_matches('.').to_string();
        let y = format!("{:.10}", self.y).trim_end_matches('0').trim_end_matches('.').to_string();
        write!(f, "Vec2 ({}, {})", x, y)
    }
}

// --- Arithmetic Implementation ---

impl Add for Vec2 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
       Self { x: self.x + rhs.x, y: self.y + rhs.y }
    }
}

impl Sub for Vec2 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self { x: self.x - rhs.x, y: self.y - rhs.y }
    }
}

impl Mul for Vec2 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self { x: self.x * rhs.x, y: self.y * rhs.y }
    }
}

impl Mul<f64> for Vec2 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: f64) -> Self::Output {
        Self { x: self.x * rhs, y: self.y * rhs }
    }
}

impl Div for Vec2 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        Self { x: self.x / rhs.x, y: self.y / rhs.y }
    }
}

impl Div<f64> for Vec2 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: f64) -> Self::Output {
        Self { x: self.x / rhs, y: self.y / rhs }
    }
}

impl Neg for Vec2 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self::Output {
        Self { x: -self.x, y: -self.y }
    }
}

impl AddAssign for Vec2 {
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
    }
}

impl SubAssign for Vec2 {
    fn sub_assign(&mut self, rhs: Self) {
        self.x -= rhs.x;
        self.y -= rhs.y;
    }
}

impl MulAssign for Vec2 {
    fn mul_assign(&mut self, rhs: Self) {
        self.x *= rhs.x;
        self.y *= rhs.y;
    }
}

impl MulAssign<f64> for Vec2 {
    #[inline]
    fn mul_assign(&mut self, rhs: f64) {
        self.x *= rhs;
        self.y *= rhs;
    }
}

impl DivAssign for Vec2 {
    fn div_assign(&mut self, rhs: Self) {
        self.x /= rhs.x;
        self.y /= rhs.y;
    }
}

impl DivAssign<f64> for Vec2 {
    #[inline]
    fn div_assign(&mut self, rhs: f64) {
        self.x /= rhs;
        self.y /= rhs;
    }
}

// --- VecOperations Implementation ---

impl VecOperations for Vec2 {
    /// Calculates the dot product of two [Vec2].
    /// 
    /// # Examples
    /// ```
    /// # use engine::math::vecs::vec2::Vec2;
    /// # use engine::math::vecs::VecOperations;
    /// let a = Vec2::new(2.0, 2.0);
    /// let b = Vec2::new(2.0, 2.0);
    /// assert_eq!(a.dot(b), 8.0);
    /// ```
    #[inline]
    fn dot(self, other: Self) -> f64 {
        self.x * other.x + self.y * other.y
    }

    /// Calculates square length of [Vec2]. (Faster than [length])
    /// 
    /// # Examples
    /// ```
    /// # use engine::math::vecs::vec2::Vec2;
    /// # use engine::math::vecs::VecOperations;
    /// let a = Vec2::new(2.0, 2.0);
    /// assert_eq!(a.length2(), 8.0);
    /// ```
    #[inline]
    fn length2(self) -> f64 {
        self.x * self.x + self.y * self.y
    }

    /// Calculates sqrt length of [Vec2].
    /// 
    /// # Examples
    /// ```
    /// # use engine::math::vecs::vec2::Vec2;
    /// # use engine::math::vecs::VecOperations;
    /// let a = Vec2::new(2.0, 2.0);
    /// assert_eq!(a.length(), 8.0f64.sqrt());
    /// ```
    #[inline]
    fn length(self) -> f64 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    /// Normalize [Vec2]. Resulting length will be 1.0.
    /// 
    /// # Examples
    /// ```
    /// # use engine::math::vecs::vec2::Vec2;
    /// # use engine::math::vecs::VecOperations;
    /// let a = Vec2::new(2.0, 2.0).norm();
    /// assert!((a.length() - 1.0).abs() < f64::EPSILON);
    /// ```
    #[inline]
    fn norm(self) -> Self {
        self / self.length()
    }

    /// Calculates the Euclidean distance between two [Vec2].
    /// 
    /// # Examples
    /// ```
    /// # use engine::math::vecs::vec2::Vec2;
    /// # use engine::math::vecs::VecOperations;
    /// let a = Vec2::new(2.0, 2.0);
    /// let b = Vec2::new(2.0, 2.0);
    /// assert_eq!(a.distance(b), 0.0);
    /// ```
    #[inline]
    fn distance(self, other: Self) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }

    /// Linear interpolation between two [Vec2] by a factor `t`.
    /// 
    /// # Examples
    /// ```
    /// # use engine::math::vecs::vec2::Vec2;
    /// # use engine::math::vecs::VecOperations;
    /// let a = Vec2::new(0.0, 0.0);
    /// let b = Vec2::new(2.0, 2.0);
    /// assert_eq!(a.lerp(b, 0.5), Vec2::new(1.0, 1.0));
    /// ```
    #[inline]
    fn lerp(self, other: Self, t: f64) -> Self {
        self * (1.0 - t) + other * t
    }

    /// Returns a [Vec2] containing the minimum components of two vectors.
    /// 
    /// # Examples
    /// ```
    /// # use engine::math::vecs::vec2::Vec2;
    /// # use engine::math::vecs::VecOperations;
    /// let a = Vec2::new(0.0, 5.0);
    /// let b = Vec2::new(1.0, 2.0);
    /// assert_eq!(a.min(b), Vec2::new(0.0, 2.0));
    /// ```
    #[inline]
    fn min(self, other: Self) -> Self {
        Self { x: self.x.min(other.x), y: self.y.min(other.y) }
    }

    /// Returns a [Vec2] containing the maximum components of two vectors.
    /// 
    /// # Examples
    /// ```
    /// # use engine::math::vecs::vec2::Vec2;
    /// # use engine::math::vecs::VecOperations;
    /// let a = Vec2::new(0.0, 5.0);
    /// let b = Vec2::new(1.0, 2.0);
    /// assert_eq!(a.max(b), Vec2::new(1.0, 5.0));
    /// ```
    #[inline]
    fn max(self, other: Self) -> Self {
        Self { x: self.x.max(other.x), y: self.y.max(other.y) }
    }

    /// Clamps the [Vec2] components between `min` and `max` vectors.
    /// 
    /// # Examples
    /// ```
    /// # use engine::math::vecs::vec2::Vec2;
    /// # use engine::math::vecs::VecOperations;
    /// let min = Vec2::new(-1.0, -1.0);
    /// let max = Vec2::new(1.0, 1.0);
    /// let v = Vec2::new(2.0, -2.0);
    /// assert_eq!(v.clamp(min, max), Vec2::new(1.0, -1.0));
    /// ```
    #[inline]
    fn clamp(self, min: Self, max: Self) -> Self {
        Self { x: self.x.clamp(min.x, max.x), y: self.y.clamp(min.y, max.y) }
    }
}


#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_vec2_display() {
        let a = Vec2 { x: 2.0, y: 3.0 };
        assert_eq!(format!("{}", a), "Vec2 (2, 3)");

        let b = Vec2 { x: 1.5, y: 0.25 };
        assert_eq!(format!("{}", b), "Vec2 (1.5, 0.25)");

        let c = Vec2 { x: 1.1234567890123, y: 0.0000000001 };
        assert_eq!(format!("{}", c), "Vec2 (1.123456789, 0.0000000001)");

        let d = Vec2 { x: 0.0, y: 0.0 };
        assert_eq!(format!("{}", d), "Vec2 (0, 0)");
    }
 
    #[test]
    fn test_vec2_add() {
        let a = Vec2 { x: 2.0, y: 2.0 };
        let b = Vec2 { x: 2.0, y: 2.0 };
        assert_eq!(a + b, Vec2 { x: 4.0, y: 4.0})
    }

    #[test]
    fn test_vec2_sub() {
        let a = Vec2 { x: 2.0, y: 2.0 };
        let b = Vec2 { x: 2.0, y: 2.0 };
        assert_eq!(a - b, Vec2 { x: 0.0, y: 0.0 })
    }

    #[test]
    fn test_vec2_mul() {
        let a = Vec2 { x: 2.0, y: 2.0 };
        let b = Vec2 { x: 2.0, y: 2.0 };
        assert_eq!(a * b, Vec2 { x: 4.0, y: 4.0 })
    }

    #[test]
    fn test_vec2_mulscalar() {
        let a = Vec2 { x: 2.0, y: 2.0 };
        assert_eq!(a * 2.0, Vec2 { x: 4.0, y: 4.0 })
    }

    #[test]
    fn test_vec2_div() {
        let a = Vec2 { x: 2.0, y: 2.0 };
        let b = Vec2 { x: 2.0, y: 2.0 };
        assert_eq!(a / b, Vec2 { x: 1.0, y: 1.0 })
    }

    #[test]
    fn test_vec2_divscalar() {
        let a = Vec2 { x: 2.0, y: 2.0 };
        assert_eq!(a / 2.0, Vec2 { x: 1.0, y: 1.0 })
    }

    #[test]
    fn test_vec2_neg() {
        let a = Vec2 { x: 2.0, y: 2.0 };
        assert_eq!(-a, Vec2 { x: -2.0, y: -2.0 })
    }

    #[test]
    fn test_vec2_addasign() {
        let mut a = Vec2 { x: 2.0, y: 2.0 };
        a += Vec2 { x: 2.0, y: 2.0 };
        assert_eq!(a, Vec2 { x: 4.0, y: 4.0 })
    }

    #[test]
    fn test_vec2_subassign() {
        let mut a = Vec2 { x: 2.0, y: 2.0 };
        a -= Vec2 { x: 2.0, y: 2.0 };
        assert_eq!(a, Vec2 { x: 0.0, y: 0.0 })
    }

    #[test]
    fn test_vec2_mulassign() {
        let mut a = Vec2 { x: 2.0, y: 2.0 };
        a *= Vec2 { x: 2.0, y: 2.0 };
        assert_eq!(a, Vec2 { x: 4.0, y: 4.0 })
    }

    #[test]
    fn test_vec2_mulassign_scalar() {
        let mut a = Vec2 { x: 2.0, y: 2.0 };
        a *= 2.0;
        assert_eq!(a, Vec2 { x: 4.0, y: 4.0 })
    }

    #[test]
    fn test_vec2_divassign() {
        let mut a = Vec2 { x: 2.0, y: 2.0 };
        a /= Vec2 { x: 2.0, y: 2.0 };
        assert_eq!(a, Vec2 { x: 1.0, y: 1.0 })
    }

    #[test]
    fn test_vec2_divassign_scalar() {
        let mut a = Vec2 { x: 2.0, y: 2.0 };
        a /= 2.0;
        assert_eq!(a, Vec2 { x: 1.0, y: 1.0 })
    }

    #[test]
    fn test_vec2_dot() {
        let a = Vec2 { x: 2.0, y: 2.0 };
        let b = Vec2 { x: 2.0, y: 2.0 };
        assert_eq!(a.dot(b), 8.0)
    }

    #[test]
    fn test_vec2_length2() {
        let a = Vec2 { x: 2.0, y: 2.0 };
        assert_eq!(a.length2(), 8.0)
    }

    #[test]
    fn test_vec2_length() {
        let a = Vec2 { x: 2.0, y: 2.0 };
        assert_eq!(a.length(), 8.0f64.sqrt())
    }   

    #[test]
    fn test_vec2_norm() {
        let a = Vec2 { x: 2.0, y: 2.0 }.norm();
        assert!((a.length() - 1.0).abs() < f64::EPSILON)
    }

    #[test]
    fn test_vec2_distance() {
        let a = Vec2 { x: 2.0, y: 2.0 };
        let b = Vec2 { x: 2.0, y: 2.0 };
        assert_eq!(a.distance(b), 0.0)
    }

    #[test]
    fn test_vec2_lerp() {
        let a = Vec2 { x: 0.0, y: 0.0 };
        let b = Vec2 { x: 2.0, y: 2.0 };
        assert_eq!(a.lerp(b, 0.5), Vec2 { x: 1.0, y: 1.0 })
    }

    #[test]
    fn test_vec2_min() {
        let a = Vec2 { x: 0.0, y: 0.0 };
        let b = Vec2 { x: 1.0, y: 1.0 };
        assert_eq!(a.min(b), Vec2 { x: 0.0, y: 0.0 })
    }

    #[test]
    fn test_vec2_max() {
        let a = Vec2 { x: 0.0, y: 0.0 };
        let b = Vec2 { x: 1.0, y: 1.0 };
        assert_eq!(a.max(b), Vec2 { x: 1.0, y: 1.0 })
    }

    #[test]
    fn test_vec2_clamp() {
        let max = Vec2 { x: 1.0, y: 1.0};
        let min = Vec2 { x: -1.0, y: -1.0 };

        let out_min = Vec2 { x: -2.0, y: -2.0 };
        let mid = Vec2 { x: 0.0, y: 0.0 };
        let out_max = Vec2 { x: 2.0, y: 2.0 };
        
        let mixed_out_min = Vec2 { x: -2.0, y: 0.0 };
        let mixed_out_max = Vec2 { x: 2.0, y: 0.0 };

        assert_eq!(out_min.clamp(min, max), Vec2 { x: -1.0, y: -1.0});
        assert_eq!(mid.clamp(min, max), Vec2 { x: 0.0, y: 0.0 });
        assert_eq!(out_max.clamp(min, max), Vec2 { x: 1.0, y: 1.0 });

        assert_eq!(mixed_out_min.clamp(min, max), Vec2 { x: -1.0, y: 0.0 });
        assert_eq!(mixed_out_max.clamp(min, max), Vec2 { x: 1.0, y: 0.0 });
    }
}