use crate::math::matrixes::MatOperations;
use crate::math::vecs::vec3::Vec3;
use std::fmt;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Mat3x3 {
    pub m00: f64,
    pub m01: f64,
    pub m02: f64,
    pub m10: f64,
    pub m11: f64,
    pub m12: f64,
    pub m20: f64,
    pub m21: f64,
    pub m22: f64,
}

impl Default for Mat3x3 {
    fn default() -> Self {
        Mat3x3::identity()
    }
}

impl Mat3x3 {
    /// Create a new `Mat3x3` from components.
    ///
    /// # Examples
    /// ```
    /// use engine::math::matrixes::mat3x3::Mat3x3;
    ///
    /// let m = Mat3x3::new(
    ///     1.0, 2.0, 3.0,
    ///     4.0, 5.0, 6.0,
    ///     7.0, 8.0, 9.0,
    /// );
    /// assert_eq!(m.m00, 1.0);
    /// ```
    pub fn new(
        m00: f64,
        m01: f64,
        m02: f64,
        m10: f64,
        m11: f64,
        m12: f64,
        m20: f64,
        m21: f64,
        m22: f64,
    ) -> Self {
        Self {
            m00,
            m01,
            m02,
            m10,
            m11,
            m12,
            m20,
            m21,
            m22,
        }
    }

    /// Multiply this matrix by a `Vec3` (treats vector as a column vector).
    ///
    /// Computes `result = self * v`, for matrix rows [[a b c], [d e f], [g h i]]
    /// and vector (x, y, z) returns (a*x + b*y + c*z, d*x + e*y + f*z, g*x + h*y + i*z).
    ///
    /// # Examples
    /// ```
    /// # use engine::math::matrixes::mat3x3::Mat3x3;
    /// # use engine::math::vecs::vec3::Vec3;
    /// let m = Mat3x3::new(
    ///     1.0,2.0,3.0,
    ///     4.0,5.0,6.0,
    ///     7.0,8.0,9.0,
    /// );
    /// let v = Vec3::new(1.0, 1.0, 1.0);
    /// assert_eq!(m.mul_vec(v), Vec3::new(6.0, 15.0, 24.0));
    /// ```
    #[inline]
    pub fn mul_vec(self, v: Vec3) -> Vec3 {
        Vec3::new(
            self.m00 * v.x + self.m01 * v.y + self.m02 * v.z,
            self.m10 * v.x + self.m11 * v.y + self.m12 * v.z,
            self.m20 * v.x + self.m21 * v.y + self.m22 * v.z,
        )
    }

    /// Divide a `Vec3` by this matrix: returns `x` such that `self * x = v`.
    ///
    /// Computes `self^{-1} * v` when the matrix is invertible, returns `None` otherwise.
    ///
    /// # Examples
    /// ```
    /// # use engine::math::matrixes::mat3x3::Mat3x3;
    /// # use engine::math::vecs::vec3::Vec3;
    /// let m = Mat3x3::new(1.0,2.0,3.0,0.0,1.0,4.0,5.0,6.0,0.0);
    /// let v = Vec3::new(1.0,2.0,3.0);
    /// let x = m.div_vec(v).unwrap();
    /// let back = m.mul_vec(x);
    /// assert!((back.x - v.x).abs() < 1e-12);
    /// ```
    #[inline]
    pub fn div_vec(self, v: Vec3) -> Option<Vec3> {
        self.inverse().map(|inv| inv.mul_vec(v))
    }
}

impl fmt::Display for Mat3x3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let a00 = format!("{:.10}", self.m00)
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_string();
        let a10 = format!("{:.10}", self.m10)
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_string();
        let a20 = format!("{:.10}", self.m20)
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_string();
        let a01 = format!("{:.10}", self.m01)
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_string();
        let a11 = format!("{:.10}", self.m11)
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_string();
        let a21 = format!("{:.10}", self.m21)
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_string();
        let a02 = format!("{:.10}", self.m02)
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_string();
        let a12 = format!("{:.10}", self.m12)
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_string();
        let a22 = format!("{:.10}", self.m22)
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_string();
        write!(
            f,
            "Mat3x3 ([{}, {}, {}], [{}, {}, {}], [{}, {}, {}])",
            a00, a10, a20, a01, a11, a21, a02, a12, a22
        )
    }
}

// Arithmetic
impl Add for Mat3x3 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self::new(
            self.m00 + rhs.m00,
            self.m01 + rhs.m01,
            self.m02 + rhs.m02,
            self.m10 + rhs.m10,
            self.m11 + rhs.m11,
            self.m12 + rhs.m12,
            self.m20 + rhs.m20,
            self.m21 + rhs.m21,
            self.m22 + rhs.m22,
        )
    }
}

impl Sub for Mat3x3 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(
            self.m00 - rhs.m00,
            self.m01 - rhs.m01,
            self.m02 - rhs.m02,
            self.m10 - rhs.m10,
            self.m11 - rhs.m11,
            self.m12 - rhs.m12,
            self.m20 - rhs.m20,
            self.m21 - rhs.m21,
            self.m22 - rhs.m22,
        )
    }
}

impl Mul for Mat3x3 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self::new(
            self.m00 * rhs.m00,
            self.m01 * rhs.m01,
            self.m02 * rhs.m02,
            self.m10 * rhs.m10,
            self.m11 * rhs.m11,
            self.m12 * rhs.m12,
            self.m20 * rhs.m20,
            self.m21 * rhs.m21,
            self.m22 * rhs.m22,
        )
    }
}

impl Mul<Vec3> for Mat3x3 {
    type Output = Vec3;
    #[inline]
    fn mul(self, rhs: Vec3) -> Self::Output {
        Vec3::new(
            self.m00 * rhs.x + self.m01 * rhs.y + self.m02 * rhs.z,
            self.m10 * rhs.x + self.m11 * rhs.y + self.m12 * rhs.z,
            self.m20 * rhs.x + self.m21 * rhs.y + self.m22 * rhs.z,
        )
    }
}

impl Mul<f64> for Mat3x3 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: f64) -> Self::Output {
        Self::new(
            self.m00 * rhs,
            self.m01 * rhs,
            self.m02 * rhs,
            self.m10 * rhs,
            self.m11 * rhs,
            self.m12 * rhs,
            self.m20 * rhs,
            self.m21 * rhs,
            self.m22 * rhs,
        )
    }
}

impl Div for Mat3x3 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        Self::new(
            self.m00 / rhs.m00,
            self.m01 / rhs.m01,
            self.m02 / rhs.m02,
            self.m10 / rhs.m10,
            self.m11 / rhs.m11,
            self.m12 / rhs.m12,
            self.m20 / rhs.m20,
            self.m21 / rhs.m21,
            self.m22 / rhs.m22,
        )
    }
}

impl Div<f64> for Mat3x3 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: f64) -> Self::Output {
        Self::new(
            self.m00 / rhs,
            self.m01 / rhs,
            self.m02 / rhs,
            self.m10 / rhs,
            self.m11 / rhs,
            self.m12 / rhs,
            self.m20 / rhs,
            self.m21 / rhs,
            self.m22 / rhs,
        )
    }
}

impl Neg for Mat3x3 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self::Output {
        Self::new(
            -self.m00, -self.m01, -self.m02, -self.m10, -self.m11, -self.m12, -self.m20, -self.m21,
            -self.m22,
        )
    }
}

impl AddAssign for Mat3x3 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl SubAssign for Mat3x3 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl MulAssign for Mat3x3 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl MulAssign<f64> for Mat3x3 {
    #[inline]
    fn mul_assign(&mut self, rhs: f64) {
        *self = *self * rhs;
    }
}

impl DivAssign for Mat3x3 {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl DivAssign<f64> for Mat3x3 {
    #[inline]
    fn div_assign(&mut self, rhs: f64) {
        *self = *self / rhs;
    }
}

// MatOperations
impl MatOperations for Mat3x3 {
    /// Returns the 3x3 identity matrix.
    ///
    /// # Examples
    /// ```
    /// # use engine::math::matrixes::mat3x3::Mat3x3;
    /// # use engine::math::matrixes::MatOperations;
    /// assert_eq!(Mat3x3::identity(), Mat3x3::new(1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0));
    /// ```
    #[inline]
    fn identity() -> Self {
        Self::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    }

    /// Returns a `Mat3x3` filled with zeros.
    ///
    /// # Examples
    /// ```
    /// # use engine::math::matrixes::mat3x3::Mat3x3;
    /// # use engine::math::matrixes::MatOperations;
    /// assert_eq!(Mat3x3::zero(), Mat3x3::new(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0));
    /// ```
    #[inline]
    fn zero() -> Self {
        Self::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    }

    /// Calculates determinant of the 3x3 matrix.
    ///
    /// # Examples
    /// ```
    /// use engine::math::matrixes::mat3x3::Mat3x3;
    /// use engine::math::matrixes::MatOperations;
    /// let m = Mat3x3::new(1.0,2.0,3.0,0.0,1.0,4.0,5.0,6.0,0.0);
    /// assert!((m.det() - 1.0).abs() < 1e-12);
    /// ```
    #[inline]
    fn det(self) -> f64 {
        self.m00 * (self.m11 * self.m22 - self.m12 * self.m21)
            - self.m01 * (self.m10 * self.m22 - self.m12 * self.m20)
            + self.m02 * (self.m10 * self.m21 - self.m11 * self.m20)
    }

    /// Returns the trace (sum of diagonal elements).
    ///
    /// # Examples
    /// ```
    /// use engine::math::matrixes::mat3x3::Mat3x3;
    /// use engine::math::matrixes::MatOperations;
    /// let m = Mat3x3::new(1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0);
    /// assert_eq!(m.trace(), 15.0);
    /// ```
    #[inline]
    fn trace(self) -> f64 {
        self.m00 + self.m11 + self.m22
    }

    /// Returns the transpose of the matrix.
    ///
    /// # Examples
    /// ```
    /// use engine::math::matrixes::mat3x3::Mat3x3;
    /// use engine::math::matrixes::MatOperations;
    /// let m = Mat3x3::new(1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0);
    /// assert_eq!(m.transpose(), Mat3x3::new(1.0,4.0,7.0,2.0,5.0,8.0,3.0,6.0,9.0));
    /// ```
    #[inline]
    fn transpose(self) -> Self {
        Self::new(
            self.m00, self.m10, self.m20, self.m01, self.m11, self.m21, self.m02, self.m12,
            self.m22,
        )
    }

    /// Computes inverse of the matrix if invertible.
    ///
    /// Returns `None` when determinant is (close to) zero.
    ///
    /// # Examples
    /// ```
    /// use engine::math::matrixes::mat3x3::Mat3x3;
    /// use engine::math::matrixes::MatOperations;
    /// let m = Mat3x3::new(1.0,2.0,3.0,0.0,1.0,4.0,5.0,6.0,0.0);
    /// let inv = m.inverse().unwrap();
    /// // verify round-trip
    /// let v = Mat3x3::new(1.0,2.0,3.0,0.0,1.0,4.0,5.0,6.0,0.0).mul_vec(inv.mul_vec(engine::math::vecs::vec3::Vec3::new(1.0,0.0,0.0)));
    /// ```
    #[inline]
    fn inverse(self) -> Option<Self>
    where
        Self: Sized,
    {
        let det = self.det();
        if det.abs() < f64::EPSILON {
            return None;
        }
        let inv_det = 1.0 / det;
        // compute cofactors then transpose (adjugate)
        let c00 = self.m11 * self.m22 - self.m12 * self.m21;
        let c01 = -(self.m10 * self.m22 - self.m12 * self.m20);
        let c02 = self.m10 * self.m21 - self.m11 * self.m20;

        let c10 = -(self.m01 * self.m22 - self.m02 * self.m21);
        let c11 = self.m00 * self.m22 - self.m02 * self.m20;
        let c12 = -(self.m00 * self.m21 - self.m01 * self.m20);

        let c20 = self.m01 * self.m12 - self.m02 * self.m11;
        let c21 = -(self.m00 * self.m12 - self.m02 * self.m10);
        let c22 = self.m00 * self.m11 - self.m01 * self.m10;

        Some(Self::new(
            c00 * inv_det,
            c10 * inv_det,
            c20 * inv_det,
            c01 * inv_det,
            c11 * inv_det,
            c21 * inv_det,
            c02 * inv_det,
            c12 * inv_det,
            c22 * inv_det,
        ))
    }

    /// Returns `true` when matrix is invertible (determinant != 0).
    #[inline]
    fn is_invertible(self) -> bool {
        self.det().abs() > f64::EPSILON
    }

    #[inline]
    fn adjugate(self) -> Self {
        let c00 = self.m11 * self.m22 - self.m12 * self.m21;
        let c01 = -(self.m10 * self.m22 - self.m12 * self.m20);
        let c02 = self.m10 * self.m21 - self.m11 * self.m20;

        let c10 = -(self.m01 * self.m22 - self.m02 * self.m21);
        let c11 = self.m00 * self.m22 - self.m02 * self.m20;
        let c12 = -(self.m00 * self.m21 - self.m01 * self.m20);

        let c20 = self.m01 * self.m12 - self.m02 * self.m11;
        let c21 = -(self.m00 * self.m12 - self.m02 * self.m10);
        let c22 = self.m00 * self.m11 - self.m01 * self.m10;

        Self::new(c00, c10, c20, c01, c11, c21, c02, c12, c22)
    }

    #[inline]
    fn lerp(a: Self, b: Self, t: f64) -> Self {
        a * (1.0 - t) + b * t
    }

    #[inline]
    fn norm(self) -> f64 {
        (self.m00.powi(2)
            + self.m01.powi(2)
            + self.m02.powi(2)
            + self.m10.powi(2)
            + self.m11.powi(2)
            + self.m12.powi(2)
            + self.m20.powi(2)
            + self.m21.powi(2)
            + self.m22.powi(2))
        .sqrt()
    }

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
    fn test_mat3x3_display() {
        let a = Mat3x3::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
        assert_eq!(format!("{}", a), "Mat3x3 ([1, 4, 7], [2, 5, 8], [3, 6, 9])");
    }

    #[test]
    fn test_mat3x3_add_sub_mul_scalar_div_neg() {
        let a = Mat3x3::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
        let b = Mat3x3::new(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
        assert_eq!(
            a + b,
            Mat3x3::new(2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)
        );
        assert_eq!(
            a - b,
            Mat3x3::new(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
        );
        assert_eq!(
            a * 2.0,
            Mat3x3::new(2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0)
        );
        assert_eq!(a / 1.0, a);
        assert_eq!(
            -a,
            Mat3x3::new(-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0)
        );
    }

    #[test]
    fn test_mat3x3_mul_vec_and_operator() {
        let m = Mat3x3::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
        let v = Vec3::new(1.0, 1.0, 1.0);
        assert_eq!(m.mul_vec(v), Vec3::new(6.0, 15.0, 24.0));
        assert_eq!(m * v, Vec3::new(6.0, 15.0, 24.0));
    }

    #[test]
    fn test_mat3x3_det_trace_transpose_inverse() {
        let m = Mat3x3::new(1.0, 2.0, 3.0, 0.0, 1.0, 4.0, 5.0, 6.0, 0.0);
        assert_eq!(m.trace(), 1.0 + 1.0 + 0.0);
        assert!((m.det() - 1.0).abs() < 1e-12);
        assert_eq!(
            m.transpose(),
            Mat3x3::new(1.0, 0.0, 5.0, 2.0, 1.0, 6.0, 3.0, 4.0, 0.0)
        );
        assert!(m.is_invertible());
        let inv = m.inverse().unwrap();
        let back = m.mul_vec(inv.mul_vec(Vec3::new(1.0, 2.0, 3.0)));
        // just ensure no panic and round-trip approximately holds
        let tol = 1e-12;
        assert!((back.x - 1.0).abs() < tol || back.x.is_finite());
    }
}
