use crate::math::matrixes::MatOperations;
use crate::math::vecs::vec4::Vec4;
use std::fmt;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Mat4x4 {
    pub m00: f64,
    pub m01: f64,
    pub m02: f64,
    pub m03: f64,
    pub m10: f64,
    pub m11: f64,
    pub m12: f64,
    pub m13: f64,
    pub m20: f64,
    pub m21: f64,
    pub m22: f64,
    pub m23: f64,
    pub m30: f64,
    pub m31: f64,
    pub m32: f64,
    pub m33: f64,
}

impl Default for Mat4x4 {
    fn default() -> Self {
        Mat4x4::identity()
    }
}

impl Mat4x4 {
    /// Create a new `Mat4x4` from components.
    pub fn new(
        m00: f64,
        m01: f64,
        m02: f64,
        m03: f64,
        m10: f64,
        m11: f64,
        m12: f64,
        m13: f64,
        m20: f64,
        m21: f64,
        m22: f64,
        m23: f64,
        m30: f64,
        m31: f64,
        m32: f64,
        m33: f64,
    ) -> Self {
        Self {
            m00,
            m01,
            m02,
            m03,
            m10,
            m11,
            m12,
            m13,
            m20,
            m21,
            m22,
            m23,
            m30,
            m31,
            m32,
            m33,
        }
    }

    /// Multiply this matrix by a `Vec4` (row-major multiplication).
    #[inline]
    pub fn mul_vec(self, v: Vec4) -> Vec4 {
        Vec4::new(
            self.m00 * v.x + self.m01 * v.y + self.m02 * v.z + self.m03 * v.w,
            self.m10 * v.x + self.m11 * v.y + self.m12 * v.z + self.m13 * v.w,
            self.m20 * v.x + self.m21 * v.y + self.m22 * v.z + self.m23 * v.w,
            self.m30 * v.x + self.m31 * v.y + self.m32 * v.z + self.m33 * v.w,
        )
    }

    /// Divide a `Vec4` by this matrix (solve self * x = v).
    #[inline]
    pub fn div_vec(self, v: Vec4) -> Option<Vec4> {
        self.inverse().map(|inv| inv.mul_vec(v))
    }
}

impl fmt::Display for Mat4x4 {
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
        let a30 = format!("{:.10}", self.m30)
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
        let a31 = format!("{:.10}", self.m31)
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
        let a32 = format!("{:.10}", self.m32)
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_string();
        let a03 = format!("{:.10}", self.m03)
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_string();
        let a13 = format!("{:.10}", self.m13)
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_string();
        let a23 = format!("{:.10}", self.m23)
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_string();
        let a33 = format!("{:.10}", self.m33)
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_string();
        write!(
            f,
            "Mat4x4 ([{}, {}, {}, {}], [{}, {}, {}, {}], [{}, {}, {}, {}], [{}, {}, {}, {}])",
            a00, a10, a20, a30, a01, a11, a21, a31, a02, a12, a22, a32, a03, a13, a23, a33
        )
    }
}

// Arithmetic elementwise
impl Add for Mat4x4 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self::new(
            self.m00 + rhs.m00,
            self.m01 + rhs.m01,
            self.m02 + rhs.m02,
            self.m03 + rhs.m03,
            self.m10 + rhs.m10,
            self.m11 + rhs.m11,
            self.m12 + rhs.m12,
            self.m13 + rhs.m13,
            self.m20 + rhs.m20,
            self.m21 + rhs.m21,
            self.m22 + rhs.m22,
            self.m23 + rhs.m23,
            self.m30 + rhs.m30,
            self.m31 + rhs.m31,
            self.m32 + rhs.m32,
            self.m33 + rhs.m33,
        )
    }
}

impl Sub for Mat4x4 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(
            self.m00 - rhs.m00,
            self.m01 - rhs.m01,
            self.m02 - rhs.m02,
            self.m03 - rhs.m03,
            self.m10 - rhs.m10,
            self.m11 - rhs.m11,
            self.m12 - rhs.m12,
            self.m13 - rhs.m13,
            self.m20 - rhs.m20,
            self.m21 - rhs.m21,
            self.m22 - rhs.m22,
            self.m23 - rhs.m23,
            self.m30 - rhs.m30,
            self.m31 - rhs.m31,
            self.m32 - rhs.m32,
            self.m33 - rhs.m33,
        )
    }
}

impl Mul for Mat4x4 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self::new(
            self.m00 * rhs.m00,
            self.m01 * rhs.m01,
            self.m02 * rhs.m02,
            self.m03 * rhs.m03,
            self.m10 * rhs.m10,
            self.m11 * rhs.m11,
            self.m12 * rhs.m12,
            self.m13 * rhs.m13,
            self.m20 * rhs.m20,
            self.m21 * rhs.m21,
            self.m22 * rhs.m22,
            self.m23 * rhs.m23,
            self.m30 * rhs.m30,
            self.m31 * rhs.m31,
            self.m32 * rhs.m32,
            self.m33 * rhs.m33,
        )
    }
}

impl Mul<Vec4> for Mat4x4 {
    type Output = Vec4;
    #[inline]
    fn mul(self, rhs: Vec4) -> Self::Output {
        Vec4::new(
            self.m00 * rhs.x + self.m01 * rhs.y + self.m02 * rhs.z + self.m03 * rhs.w,
            self.m10 * rhs.x + self.m11 * rhs.y + self.m12 * rhs.z + self.m13 * rhs.w,
            self.m20 * rhs.x + self.m21 * rhs.y + self.m22 * rhs.z + self.m23 * rhs.w,
            self.m30 * rhs.x + self.m31 * rhs.y + self.m32 * rhs.z + self.m33 * rhs.w,
        )
    }
}

impl Mul<f64> for Mat4x4 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: f64) -> Self::Output {
        Self::new(
            self.m00 * rhs,
            self.m01 * rhs,
            self.m02 * rhs,
            self.m03 * rhs,
            self.m10 * rhs,
            self.m11 * rhs,
            self.m12 * rhs,
            self.m13 * rhs,
            self.m20 * rhs,
            self.m21 * rhs,
            self.m22 * rhs,
            self.m23 * rhs,
            self.m30 * rhs,
            self.m31 * rhs,
            self.m32 * rhs,
            self.m33 * rhs,
        )
    }
}

impl Div for Mat4x4 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        Self::new(
            self.m00 / rhs.m00,
            self.m01 / rhs.m01,
            self.m02 / rhs.m02,
            self.m03 / rhs.m03,
            self.m10 / rhs.m10,
            self.m11 / rhs.m11,
            self.m12 / rhs.m12,
            self.m13 / rhs.m13,
            self.m20 / rhs.m20,
            self.m21 / rhs.m21,
            self.m22 / rhs.m22,
            self.m23 / rhs.m23,
            self.m30 / rhs.m30,
            self.m31 / rhs.m31,
            self.m32 / rhs.m32,
            self.m33 / rhs.m33,
        )
    }
}

impl Div<f64> for Mat4x4 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: f64) -> Self::Output {
        Self::new(
            self.m00 / rhs,
            self.m01 / rhs,
            self.m02 / rhs,
            self.m03 / rhs,
            self.m10 / rhs,
            self.m11 / rhs,
            self.m12 / rhs,
            self.m13 / rhs,
            self.m20 / rhs,
            self.m21 / rhs,
            self.m22 / rhs,
            self.m23 / rhs,
            self.m30 / rhs,
            self.m31 / rhs,
            self.m32 / rhs,
            self.m33 / rhs,
        )
    }
}

impl Neg for Mat4x4 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self::Output {
        Self::new(
            -self.m00, -self.m01, -self.m02, -self.m03, -self.m10, -self.m11, -self.m12, -self.m13,
            -self.m20, -self.m21, -self.m22, -self.m23, -self.m30, -self.m31, -self.m32, -self.m33,
        )
    }
}

impl AddAssign for Mat4x4 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}
impl SubAssign for Mat4x4 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}
impl MulAssign for Mat4x4 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}
impl MulAssign<f64> for Mat4x4 {
    #[inline]
    fn mul_assign(&mut self, rhs: f64) {
        *self = *self * rhs;
    }
}
impl DivAssign for Mat4x4 {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}
impl DivAssign<f64> for Mat4x4 {
    #[inline]
    fn div_assign(&mut self, rhs: f64) {
        *self = *self / rhs;
    }
}

// Helpers for determinant and cofactors
fn det3(
    a00: f64,
    a01: f64,
    a02: f64,
    a10: f64,
    a11: f64,
    a12: f64,
    a20: f64,
    a21: f64,
    a22: f64,
) -> f64 {
    a00 * (a11 * a22 - a12 * a21) - a01 * (a10 * a22 - a12 * a20) + a02 * (a10 * a21 - a11 * a20)
}

// MatOperations
impl MatOperations for Mat4x4 {
    /// Returns the 4x4 identity matrix.
    ///
    /// # Examples
    /// ```
    /// # use engine::math::matrixes::mat4x4::Mat4x4;
    /// # use engine::math::matrixes::MatOperations;
    /// assert_eq!(Mat4x4::identity(), Mat4x4::new(
    ///     1.0,0.0,0.0,0.0,
    ///     0.0,1.0,0.0,0.0,
    ///     0.0,0.0,1.0,0.0,
    ///     0.0,0.0,0.0,1.0,
    /// ));
    /// ```
    #[inline]
    fn identity() -> Self {
        Self::new(
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        )
    }

    /// Returns a `Mat4x4` filled with zeros.
    ///
    /// # Examples
    /// ```
    /// # use engine::math::matrixes::mat4x4::Mat4x4;
    /// # use engine::math::matrixes::MatOperations;
    /// assert_eq!(Mat4x4::zero(), Mat4x4::new(
    ///     0.0,0.0,0.0,0.0,
    ///     0.0,0.0,0.0,0.0,
    ///     0.0,0.0,0.0,0.0,
    ///     0.0,0.0,0.0,0.0,
    /// ));
    /// ```
    #[inline]
    fn zero() -> Self {
        Self::new(
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        )
    }

    /// Calculates determinant of the 4x4 matrix.
    ///
    /// # Examples
    /// ```
    /// use engine::math::matrixes::mat4x4::Mat4x4;
    /// use engine::math::matrixes::MatOperations;
    /// let m = Mat4x4::identity();
    /// assert_eq!(m.det(), 1.0);
    /// ```
    #[inline]
    fn det(self) -> f64 {
        // Expand along first row using 3x3 minors
        let m = &self;
        m.m00
            * det3(
                m.m11, m.m12, m.m13, m.m21, m.m22, m.m23, m.m31, m.m32, m.m33,
            )
            - m.m01
                * det3(
                    m.m10, m.m12, m.m13, m.m20, m.m22, m.m23, m.m30, m.m32, m.m33,
                )
            + m.m02
                * det3(
                    m.m10, m.m11, m.m13, m.m20, m.m21, m.m23, m.m30, m.m31, m.m33,
                )
            - m.m03
                * det3(
                    m.m10, m.m11, m.m12, m.m20, m.m21, m.m22, m.m30, m.m31, m.m32,
                )
    }

    /// Returns the trace (sum of diagonal elements).
    ///
    /// # Examples
    /// ```
    /// use engine::math::matrixes::mat4x4::Mat4x4;
    /// use engine::math::matrixes::MatOperations;
    /// let m = Mat4x4::new(
    ///     1.0,2.0,3.0,4.0,
    ///     5.0,6.0,7.0,8.0,
    ///     9.0,10.0,11.0,12.0,
    ///     13.0,14.0,15.0,16.0,
    /// );
    /// assert_eq!(m.trace(), 34.0);
    /// ```
    #[inline]
    fn trace(self) -> f64 {
        self.m00 + self.m11 + self.m22 + self.m33
    }

    /// Returns the transpose of the matrix.
    ///
    /// # Examples
    /// ```
    /// use engine::math::matrixes::mat4x4::Mat4x4;
    /// use engine::math::matrixes::MatOperations;
    /// let m = Mat4x4::new(
    ///     1.0,2.0,3.0,4.0,
    ///     5.0,6.0,7.0,8.0,
    ///     9.0,10.0,11.0,12.0,
    ///     13.0,14.0,15.0,16.0,
    /// );
    /// let t = m.transpose();
    /// assert_eq!(t.m01, 5.0);
    /// ```
    #[inline]
    fn transpose(self) -> Self {
        Self::new(
            self.m00, self.m10, self.m20, self.m30, self.m01, self.m11, self.m21, self.m31,
            self.m02, self.m12, self.m22, self.m32, self.m03, self.m13, self.m23, self.m33,
        )
    }

    /// Computes inverse of the matrix if invertible.
    ///
    /// Returns `None` when determinant is (close to) zero.
    ///
    /// # Examples
    /// ```
    /// use engine::math::matrixes::mat4x4::Mat4x4;
    /// use engine::math::matrixes::MatOperations;
    /// let m = Mat4x4::identity();
    /// assert!(m.inverse().is_some());
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
        let m = &self;
        // compute cofactor matrix (C_ij = (-1)^{i+j} det(Minor_{ij}))
        let mut c = [0.0f64; 16];
        for i in 0..4 {
            for j in 0..4 {
                // build 3x3 minor excluding row i and col j
                let mut idx = 0usize;
                let mut minor = [0.0f64; 9];
                for r in 0..4 {
                    if r == i {
                        continue;
                    }
                    for cc in 0..4 {
                        if cc == j {
                            continue;
                        }
                        let val = match (r, cc) {
                            (0, 0) => m.m00,
                            (0, 1) => m.m01,
                            (0, 2) => m.m02,
                            (0, 3) => m.m03,
                            (1, 0) => m.m10,
                            (1, 1) => m.m11,
                            (1, 2) => m.m12,
                            (1, 3) => m.m13,
                            (2, 0) => m.m20,
                            (2, 1) => m.m21,
                            (2, 2) => m.m22,
                            (2, 3) => m.m23,
                            (3, 0) => m.m30,
                            (3, 1) => m.m31,
                            (3, 2) => m.m32,
                            (3, 3) => m.m33,
                            _ => 0.0,
                        };
                        minor[idx] = val;
                        idx += 1;
                    }
                }
                let d = det3(
                    minor[0], minor[1], minor[2], minor[3], minor[4], minor[5], minor[6], minor[7],
                    minor[8],
                );
                let sign = if ((i + j) % 2) == 0 { 1.0 } else { -1.0 };
                c[i * 4 + j] = sign * d;
            }
        }
        // adjugate is transpose of cofactor matrix
        let adj = Self::new(
            c[0] * inv_det,
            c[4] * inv_det,
            c[8] * inv_det,
            c[12] * inv_det,
            c[1] * inv_det,
            c[5] * inv_det,
            c[9] * inv_det,
            c[13] * inv_det,
            c[2] * inv_det,
            c[6] * inv_det,
            c[10] * inv_det,
            c[14] * inv_det,
            c[3] * inv_det,
            c[7] * inv_det,
            c[11] * inv_det,
            c[15] * inv_det,
        );
        Some(adj)
    }

    /// Returns `true` when matrix is invertible (determinant != 0).
    #[inline]
    fn is_invertible(self) -> bool {
        self.det().abs() > f64::EPSILON
    }

    #[inline]
    fn adjugate(self) -> Self {
        // compute cofactor matrix then transpose
        let m = &self;
        let mut c = [0.0f64; 16];
        for i in 0..4 {
            for j in 0..4 {
                let mut idx = 0usize;
                let mut minor = [0.0f64; 9];
                for r in 0..4 {
                    if r == i {
                        continue;
                    }
                    for cc in 0..4 {
                        if cc == j {
                            continue;
                        }
                        let val = match (r, cc) {
                            (0, 0) => m.m00,
                            (0, 1) => m.m01,
                            (0, 2) => m.m02,
                            (0, 3) => m.m03,
                            (1, 0) => m.m10,
                            (1, 1) => m.m11,
                            (1, 2) => m.m12,
                            (1, 3) => m.m13,
                            (2, 0) => m.m20,
                            (2, 1) => m.m21,
                            (2, 2) => m.m22,
                            (2, 3) => m.m23,
                            (3, 0) => m.m30,
                            (3, 1) => m.m31,
                            (3, 2) => m.m32,
                            (3, 3) => m.m33,
                            _ => 0.0,
                        };
                        minor[idx] = val;
                        idx += 1;
                    }
                }
                let d = det3(
                    minor[0], minor[1], minor[2], minor[3], minor[4], minor[5], minor[6], minor[7],
                    minor[8],
                );
                let sign = if ((i + j) % 2) == 0 { 1.0 } else { -1.0 };
                c[i * 4 + j] = sign * d;
            }
        }
        Self::new(
            c[0], c[4], c[8], c[12], c[1], c[5], c[9], c[13], c[2], c[6], c[10], c[14], c[3], c[7],
            c[11], c[15],
        )
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
            + self.m03.powi(2)
            + self.m10.powi(2)
            + self.m11.powi(2)
            + self.m12.powi(2)
            + self.m13.powi(2)
            + self.m20.powi(2)
            + self.m21.powi(2)
            + self.m22.powi(2)
            + self.m23.powi(2)
            + self.m30.powi(2)
            + self.m31.powi(2)
            + self.m32.powi(2)
            + self.m33.powi(2))
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
    fn test_mat4x4_display() {
        let m = Mat4x4::new(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        );
        assert_eq!(
            format!("{}", m),
            "Mat4x4 ([1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15], [4, 8, 12, 16])"
        );
    }

    #[test]
    fn test_mat4x4_basic_ops_and_vec() {
        let a = Mat4x4::new(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        );
        let b = Mat4x4::new(
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        );
        assert_eq!(
            a + b,
            Mat4x4::new(
                2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
                17.0
            )
        );
        let v = Vec4::new(1.0, 1.0, 1.0, 1.0);
        assert_eq!(a.mul_vec(v), Vec4::new(10.0, 26.0, 42.0, 58.0));
        assert_eq!(a * v, Vec4::new(10.0, 26.0, 42.0, 58.0));
    }

    #[test]
    fn test_mat4x4_det_transpose_inverse() {
        let m = Mat4x4::identity();
        assert!(m.is_invertible());
        assert_eq!(m.det(), 1.0);
        assert_eq!(m.transpose(), m);
        assert_eq!(m.inverse().unwrap(), m);
    }
}
