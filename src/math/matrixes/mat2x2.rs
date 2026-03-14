use crate::math::matrixes::MatOperations;
use crate::math::vecs::vec2::Vec2;
use std::fmt;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Mat2x2 {
    pub m00: f64,
    pub m01: f64,
    pub m10: f64,
    pub m11: f64,
}

impl Mat2x2 {
    /// Creates a new `Mat2x2` from components.
    ///
    /// # Examples
    /// ```
    /// use engine::math::matrixes::mat2x2::Mat2x2;
    ///
    /// let m = Mat2x2::new(1.0, 2.0, 3.0, 4.0);
    /// assert_eq!(m.m00, 1.0);
    /// assert_eq!(m.m01, 2.0);
    /// assert_eq!(m.m10, 3.0);
    /// assert_eq!(m.m11, 4.0);
    /// ```
    pub fn new(m00: f64, m01: f64, m10: f64, m11: f64) -> Self {
        Self { m00, m01, m10, m11 }
    }

    /// Multiply this matrix by a `Vec2` (treating the vector as a column vector).
    ///
    /// Computes `result = self * v`, where for matrix [[a, b], [c, d]] and
    /// vector (x, y) the result is (a*x + b*y, c*x + d*y).
    ///
    /// # Examples
    /// ```
    /// # use engine::math::matrixes::mat2x2::Mat2x2;
    /// # use engine::math::vecs::vec2::Vec2;
    /// let m = Mat2x2::new(1.0, 2.0, 3.0, 4.0);
    /// let v = Vec2::new(1.0, 1.0);
    /// assert_eq!(m.mul_vec(v), Vec2::new(3.0, 7.0));
    /// ```
    #[inline]
    pub fn mul_vec(self, v: Vec2) -> Vec2 {
        Vec2::new(
            self.m00 * v.x + self.m01 * v.y,
            self.m10 * v.x + self.m11 * v.y,
        )
    }

    /// Divide a `Vec2` by this matrix: returns `x` such that `self * x = v`.
    ///
    /// This computes `self^{-1} * v` when the matrix is invertible.
    /// Returns `None` if the matrix is not invertible.
    ///
    /// # Examples
    /// ```
    /// # use engine::math::matrixes::mat2x2::Mat2x2;
    /// # use engine::math::vecs::vec2::Vec2;
    /// let m = Mat2x2::new(1.0, 2.0, 3.0, 4.0);
    /// let v = Vec2::new(3.0, 7.0);
    /// let x = m.div_vec(v).unwrap();
    /// assert!((m.mul_vec(x).x - v.x).abs() < f64::EPSILON);
    /// assert!((m.mul_vec(x).y - v.y).abs() < f64::EPSILON);
    /// ```
    #[inline]
    pub fn div_vec(self, v: Vec2) -> Option<Vec2> {
        self.inverse().map(|inv| inv.mul_vec(v))
    }
}

impl Default for Mat2x2 {
    fn default() -> Self {
        Mat2x2::identity()
    }
}

impl fmt::Display for Mat2x2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let m00 = format!("{:.10}", self.m00)
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_string();
        let m10 = format!("{:.10}", self.m10)
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_string();
        let m01 = format!("{:.10}", self.m01)
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_string();
        let m11 = format!("{:.10}", self.m11)
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_string();
        write!(f, "Mat2x2 ([{}, {}], [{}, {}])", m00, m10, m01, m11)
    }
}

// --- Arithmetic Implementation ---
impl Add for Mat2x2 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            m00: self.m00 + rhs.m00,
            m01: self.m01 + rhs.m01,
            m10: self.m10 + rhs.m10,
            m11: self.m11 + rhs.m11,
        }
    }
}

impl Sub for Mat2x2 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            m00: self.m00 - rhs.m00,
            m01: self.m01 - rhs.m01,
            m10: self.m10 - rhs.m10,
            m11: self.m11 - rhs.m11,
        }
    }
}

impl Mul for Mat2x2 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            m00: self.m00 * rhs.m00,
            m01: self.m01 * rhs.m01,
            m10: self.m10 * rhs.m10,
            m11: self.m11 * rhs.m11,
        }
    }
}

impl Mul<Vec2> for Mat2x2 {
    type Output = Vec2;
    #[inline]
    fn mul(self, rhs: Vec2) -> Self::Output {
        Vec2::new(
            self.m00 * rhs.x + self.m01 * rhs.y,
            self.m10 * rhs.x + self.m11 * rhs.y,
        )
    }
}

impl Mul<f64> for Mat2x2 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: f64) -> Self::Output {
        Self {
            m00: self.m00 * rhs,
            m01: self.m01 * rhs,
            m10: self.m10 * rhs,
            m11: self.m11 * rhs,
        }
    }
}

impl Div for Mat2x2 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        Self {
            m00: self.m00 / rhs.m00,
            m01: self.m01 / rhs.m01,
            m10: self.m10 / rhs.m10,
            m11: self.m11 / rhs.m11,
        }
    }
}

impl Div<f64> for Mat2x2 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: f64) -> Self::Output {
        Self {
            m00: self.m00 / rhs,
            m01: self.m01 / rhs,
            m10: self.m10 / rhs,
            m11: self.m11 / rhs,
        }
    }
}

impl Neg for Mat2x2 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self::Output {
        Self {
            m00: -self.m00,
            m01: -self.m01,
            m10: -self.m10,
            m11: -self.m11,
        }
    }
}

impl AddAssign for Mat2x2 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.m00 += rhs.m00;
        self.m01 += rhs.m01;
        self.m10 += rhs.m10;
        self.m11 += rhs.m11;
    }
}

impl SubAssign for Mat2x2 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.m00 -= rhs.m00;
        self.m01 -= rhs.m01;
        self.m10 -= rhs.m10;
        self.m11 -= rhs.m11;
    }
}

impl MulAssign for Mat2x2 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        self.m00 *= rhs.m00;
        self.m01 *= rhs.m01;
        self.m10 *= rhs.m10;
        self.m11 *= rhs.m11;
    }
}

impl MulAssign<f64> for Mat2x2 {
    #[inline]
    fn mul_assign(&mut self, rhs: f64) {
        self.m00 *= rhs;
        self.m01 *= rhs;
        self.m10 *= rhs;
        self.m11 *= rhs;
    }
}

impl DivAssign for Mat2x2 {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        self.m00 /= rhs.m00;
        self.m01 /= rhs.m01;
        self.m10 /= rhs.m10;
        self.m11 /= rhs.m11;
    }
}

impl DivAssign<f64> for Mat2x2 {
    #[inline]
    fn div_assign(&mut self, rhs: f64) {
        self.m00 /= rhs;
        self.m01 /= rhs;
        self.m10 /= rhs;
        self.m11 /= rhs;
    }
}

// --- MatOperations Implementation ---

impl MatOperations for Mat2x2 {
    /// Returns the 2x2 identity matrix.
    ///
    /// # Examples
    /// ```
    /// # use engine::math::matrixes::mat2x2::Mat2x2;
    /// # use engine::math::matrixes::MatOperations;
    /// let id = Mat2x2::identity();
    /// assert_eq!(id, Mat2x2::new(1.0, 0.0, 0.0, 1.0));
    /// ```
    #[inline]
    fn identity() -> Self {
        Self {
            m00: 1.0,
            m01: 0.0,
            m10: 0.0,
            m11: 1.0,
        }
    }

    /// Returns a `Mat2x2` filled with zeros.
    ///
    /// # Examples
    /// ```
    /// # use engine::math::matrixes::mat2x2::Mat2x2;
    /// # use engine::math::matrixes::MatOperations;
    /// let z = Mat2x2::zero();
    /// assert_eq!(z, Mat2x2::new(0.0, 0.0, 0.0, 0.0));
    /// ```
    #[inline]
    fn zero() -> Self {
        Self {
            m00: 0.0,
            m01: 0.0,
            m10: 0.0,
            m11: 0.0,
        }
    }

    /// Calculates determinant of the matrix.
    ///
    /// For matrix [[a, b], [c, d]] returns a * d - b * c.
    ///
    /// # Examples
    /// ```
    /// # use engine::math::matrixes::mat2x2::Mat2x2;
    /// # use engine::math::matrixes::MatOperations;
    /// let m = Mat2x2::new(1.0, 2.0, 3.0, 4.0);
    /// assert_eq!(m.det(), -2.0);
    /// ```
    #[inline]
    fn det(self) -> f64 {
        self.m00 * self.m11 - self.m01 * self.m10
    }

    /// Returns the trace (sum of diagonal elements).
    ///
    /// # Examples
    /// ```
    /// # use engine::math::matrixes::mat2x2::Mat2x2;
    /// # use engine::math::matrixes::MatOperations;
    /// let m = Mat2x2::new(1.0, 2.0, 3.0, 4.0);
    /// assert_eq!(m.trace(), 5.0);
    /// ```
    #[inline]
    fn trace(self) -> f64 {
        self.m00 + self.m11
    }

    /// Returns the transpose of the matrix.
    ///
    /// # Examples
    /// ```
    /// # use engine::math::matrixes::mat2x2::Mat2x2;
    /// # use engine::math::matrixes::MatOperations;
    /// let m = Mat2x2::new(1.0, 2.0, 3.0, 4.0);
    /// assert_eq!(m.transpose(), Mat2x2::new(1.0, 3.0, 2.0, 4.0));
    /// ```
    #[inline]
    fn transpose(self) -> Self {
        Self {
            m00: self.m00,
            m01: self.m10,
            m10: self.m01,
            m11: self.m11,
        }
    }

    /// Computes inverse of the matrix if invertible.
    ///
    /// Returns `None` when determinant is (close to) zero.
    ///
    /// # Examples
    /// ```
    /// # use engine::math::matrixes::mat2x2::Mat2x2;
    /// # use engine::math::matrixes::MatOperations;
    /// let m = Mat2x2::new(1.0, 2.0, 3.0, 4.0);
    /// let inv = m.inverse().unwrap();
    /// assert_eq!(inv, Mat2x2::new(-2.0, 1.0, 1.5, -0.5));
    /// ```
    #[inline]
    fn inverse(self) -> Option<Self>
    where
        Self: Sized,
    {
        let det = self.det();
        if det.abs() < f64::EPSILON {
            return None;
        };
        let inv_det = 1.0 / det;
        Some(Self {
            m00: self.m11 * inv_det,
            m01: -self.m01 * inv_det,
            m10: -self.m10 * inv_det,
            m11: self.m00 * inv_det,
        })
    }

    /// Returns `true` when matrix is invertible (determinant != 0).
    ///
    /// # Examples
    /// ```
    /// # use engine::math::matrixes::mat2x2::Mat2x2;
    /// # use engine::math::matrixes::MatOperations;
    /// assert!(Mat2x2::new(1.0, 2.0, 3.0, 4.0).is_invertible());
    /// assert!(!Mat2x2::new(1.0, 2.0, 2.0, 4.0).is_invertible());
    /// ```
    #[inline]
    fn is_invertible(self) -> bool {
        self.det().abs() > f64::EPSILON
    }

    /// Returns the adjugate (classical adjoint) of the matrix.
    ///
    /// # Examples
    /// ```
    /// # use engine::math::matrixes::mat2x2::Mat2x2;
    /// # use engine::math::matrixes::MatOperations;
    /// let m = Mat2x2::new(1.0, 2.0, 3.0, 4.0);
    /// assert_eq!(m.adjugate(), Mat2x2::new(4.0, -2.0, -3.0, 1.0));
    /// ```
    #[inline]
    fn adjugate(self) -> Self {
        Self {
            m00: self.m11,
            m01: -self.m01,
            m10: -self.m10,
            m11: self.m00,
        }
    }

    /// Linear interpolation between two matrices.
    ///
    /// # Examples
    /// ```
    /// # use engine::math::matrixes::mat2x2::Mat2x2;
    /// # use engine::math::matrixes::MatOperations;
    /// let a = Mat2x2::zero();
    /// let b = Mat2x2::new(2.0, 2.0, 2.0, 2.0);
    /// assert_eq!(Mat2x2::lerp(a, b, 0.5), Mat2x2::new(1.0, 1.0, 1.0, 1.0));
    /// ```
    #[inline]
    fn lerp(a: Self, b: Self, t: f64) -> Self {
        a * (1.0 - t) + b * t
    }

    /// Frobenius norm (sqrt of sum of squares of elements).
    ///
    /// # Examples
    /// ```
    /// # use engine::math::matrixes::mat2x2::Mat2x2;
    /// # use engine::math::matrixes::MatOperations;
    /// let m = Mat2x2::new(1.0, 2.0, 3.0, 4.0);
    /// let expected = (1.0f64.powi(2) + 2.0f64.powi(2) + 3.0f64.powi(2) + 4.0f64.powi(2)).sqrt();
    /// assert_eq!(m.norm(), expected);
    /// ```
    #[inline]
    fn norm(self) -> f64 {
        (self.m00.powi(2) + self.m01.powi(2) + self.m10.powi(2) + self.m11.powi(2)).sqrt()
    }

    /// Returns the matrix scaled to unit Frobenius norm.
    ///
    /// # Examples
    /// ```
    /// # use engine::math::matrixes::mat2x2::Mat2x2;
    /// # use engine::math::matrixes::MatOperations;
    /// let m = Mat2x2::new(1.0, 2.0, 3.0, 4.0);
    /// let n = m.norm();
    /// let normalized = m.normalize();
    /// let back = normalized * n;
    /// assert!((back.m00 - m.m00).abs() < f64::EPSILON);
    /// ```
    #[inline]
    fn normalize(self) -> Self {
        let n = self.norm();
        if n > f64::EPSILON { self / n } else { self }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_mat2x2_display() {
        let a = Mat2x2 {
            m00: 2.0,
            m01: 3.0,
            m10: 4.0,
            m11: 5.0,
        };
        assert_eq!(format!("{}", a), "Mat2x2 ([2, 4], [3, 5])");

        let b = Mat2x2 {
            m00: 1.5,
            m01: 0.25,
            m10: -0.5,
            m11: 0.125,
        };
        assert_eq!(format!("{}", b), "Mat2x2 ([1.5, -0.5], [0.25, 0.125])");

        let c = Mat2x2 {
            m00: 1.1234567890123,
            m01: 0.0000000001,
            m10: 0.0000000002,
            m11: 0.0000000003,
        };
        assert_eq!(
            format!("{}", c),
            "Mat2x2 ([1.123456789, 0.0000000002], [0.0000000001, 0.0000000003])"
        );

        let d = Mat2x2 {
            m00: 0.0,
            m01: 0.0,
            m10: 0.0,
            m11: 0.0,
        };
        assert_eq!(format!("{}", d), "Mat2x2 ([0, 0], [0, 0])");
    }

    #[test]
    fn test_mat2x2_add() {
        let a = Mat2x2::new(1.0, 2.0, 3.0, 4.0);
        let b = Mat2x2::new(1.0, 1.0, 1.0, 1.0);
        assert_eq!(a + b, Mat2x2::new(2.0, 3.0, 4.0, 5.0));
    }

    #[test]
    fn test_mat2x2_sub() {
        let a = Mat2x2::new(2.0, 3.0, 4.0, 5.0);
        let b = Mat2x2::new(1.0, 1.0, 1.0, 1.0);
        assert_eq!(a - b, Mat2x2::new(1.0, 2.0, 3.0, 4.0));
    }

    #[test]
    fn test_mat2x2_mul_elementwise() {
        let a = Mat2x2::new(2.0, 3.0, 4.0, 5.0);
        let b = Mat2x2::new(2.0, 2.0, 2.0, 2.0);
        assert_eq!(a * b, Mat2x2::new(4.0, 6.0, 8.0, 10.0));
    }

    #[test]
    fn test_mat2x2_mul_scalar() {
        let a = Mat2x2::new(2.0, 3.0, 4.0, 5.0);
        assert_eq!(a * 2.0, Mat2x2::new(4.0, 6.0, 8.0, 10.0));
    }

    #[test]
    fn test_mat2x2_div_elementwise() {
        let a = Mat2x2::new(4.0, 6.0, 8.0, 10.0);
        let b = Mat2x2::new(2.0, 3.0, 4.0, 5.0);
        assert_eq!(a / b, Mat2x2::new(2.0, 2.0, 2.0, 2.0));
    }

    #[test]
    fn test_mat2x2_div_scalar() {
        let a = Mat2x2::new(4.0, 6.0, 8.0, 10.0);
        assert_eq!(a / 2.0, Mat2x2::new(2.0, 3.0, 4.0, 5.0));
    }

    #[test]
    fn test_mat2x2_neg() {
        let a = Mat2x2::new(1.0, -2.0, 3.0, -4.0);
        assert_eq!(-a, Mat2x2::new(-1.0, 2.0, -3.0, 4.0));
    }

    #[test]
    fn test_mat2x2_addassign() {
        let mut a = Mat2x2::new(1.0, 1.0, 1.0, 1.0);
        a += Mat2x2::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(a, Mat2x2::new(2.0, 3.0, 4.0, 5.0));
    }

    #[test]
    fn test_mat2x2_subassign() {
        let mut a = Mat2x2::new(2.0, 3.0, 4.0, 5.0);
        a -= Mat2x2::new(1.0, 1.0, 1.0, 1.0);
        assert_eq!(a, Mat2x2::new(1.0, 2.0, 3.0, 4.0));
    }

    #[test]
    fn test_mat2x2_mulassign_elementwise_and_scalar() {
        let mut a = Mat2x2::new(2.0, 3.0, 4.0, 5.0);
        a *= Mat2x2::new(2.0, 2.0, 2.0, 2.0);
        assert_eq!(a, Mat2x2::new(4.0, 6.0, 8.0, 10.0));
        a *= 0.5;
        assert_eq!(a, Mat2x2::new(2.0, 3.0, 4.0, 5.0));
    }

    #[test]
    fn test_mat2x2_divassign_elementwise_and_scalar() {
        let mut a = Mat2x2::new(4.0, 6.0, 8.0, 10.0);
        a /= Mat2x2::new(2.0, 3.0, 4.0, 5.0);
        assert_eq!(a, Mat2x2::new(2.0, 2.0, 2.0, 2.0));
        a /= 2.0;
        assert_eq!(a, Mat2x2::new(1.0, 1.0, 1.0, 1.0));
    }

    #[test]
    fn test_mat2x2_det_trace() {
        let m = Mat2x2::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(m.det(), -2.0);
        assert_eq!(m.trace(), 5.0);
    }

    #[test]
    fn test_mat2x2_transpose() {
        let m = Mat2x2::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(m.transpose(), Mat2x2::new(1.0, 3.0, 2.0, 4.0));
    }

    #[test]
    fn test_mat2x2_inverse_and_invertible() {
        let m = Mat2x2::new(1.0, 2.0, 3.0, 4.0);
        assert!(m.is_invertible());
        let inv = m.inverse().unwrap();
        let expected = Mat2x2::new(-2.0, 1.0, 1.5, -0.5);
        assert_eq!(inv, expected);
    }

    #[test]
    fn test_mat2x2_adjugate() {
        let m = Mat2x2::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(m.adjugate(), Mat2x2::new(4.0, -2.0, -3.0, 1.0));
    }

    #[test]
    fn test_mat2x2_lerp_norm_normalize() {
        let a = Mat2x2::zero();
        let b = Mat2x2::new(2.0, 2.0, 2.0, 2.0);
        assert_eq!(Mat2x2::lerp(a, b, 0.5), Mat2x2::new(1.0, 1.0, 1.0, 1.0));

        let m = Mat2x2::new(1.0, 2.0, 3.0, 4.0);
        let n = m.norm();
        let expected_norm =
            (1.0f64.powi(2) + 2.0f64.powi(2) + 3.0f64.powi(2) + 4.0f64.powi(2)).sqrt();
        assert_eq!(n, expected_norm);

        let normalized = m.normalize();
        let back = normalized * n;
        let tol = 1e-12;
        assert!((back.m00 - m.m00).abs() < tol);
        assert!((back.m01 - m.m01).abs() < tol);
        assert!((back.m10 - m.m10).abs() < tol);
        assert!((back.m11 - m.m11).abs() < tol);
    }

    #[test]
    fn test_mat2x2_mul_vec() {
        use crate::math::vecs::vec2::Vec2;
        let m = Mat2x2::new(1.0, 2.0, 3.0, 4.0);
        let v = Vec2::new(1.0, 1.0);
        assert_eq!(m.mul_vec(v), Vec2::new(3.0, 7.0));
    }

    #[test]
    fn test_mat2x2_mul_operator_vec() {
        use crate::math::vecs::vec2::Vec2;
        let m = Mat2x2::new(1.0, 2.0, 3.0, 4.0);
        let v = Vec2::new(1.0, 1.0);
        assert_eq!(m * v, Vec2::new(3.0, 7.0));
    }

    #[test]
    fn test_mat2x2_div_vec() {
        use crate::math::vecs::vec2::Vec2;
        let m = Mat2x2::new(1.0, 2.0, 3.0, 4.0);
        let v = Vec2::new(3.0, 7.0);
        let x = m.div_vec(v).unwrap();
        let back = m.mul_vec(x);
        assert!((back.x - v.x).abs() < f64::EPSILON);
        assert!((back.y - v.y).abs() < f64::EPSILON);
    }

    #[test]
    fn test_mat2x2_identity_zero() {
        let id = Mat2x2::identity();
        assert_eq!(id, Mat2x2::new(1.0, 0.0, 0.0, 1.0));
        let z = Mat2x2::zero();
        assert_eq!(z, Mat2x2::new(0.0, 0.0, 0.0, 0.0));
    }

    #[test]
    fn test_mat2x2_inverse_none_and_is_invertible_false() {
        let m = Mat2x2::new(1.0, 2.0, 2.0, 4.0);
        assert!(!m.is_invertible());
        assert!(m.inverse().is_none());
    }

    #[test]
    fn test_mat2x2_normalize_zero() {
        let z = Mat2x2::zero();
        let normalized = z.normalize();
        assert_eq!(normalized, z);
    }
}
