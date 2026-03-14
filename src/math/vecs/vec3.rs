use crate::math::vecs::VecOperations;
use std::fmt;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// A three-dimensional vector. Perfect for describing a point in 3D space.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Default)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    /// Creating new vector [Vec3].
    ///
    /// # Examples
    /// ```
    /// use engine::math::vecs::vec3::Vec3;
    ///
    /// let vector3 = Vec3::new(1.0, 2.0, 3.0);
    /// assert_eq!(vector3.x, 1.0);
    /// assert_eq!(vector3.y, 2.0);
    /// assert_eq!(vector3.z, 3.0);
    /// ```
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Vec3 { x, y, z }
    }

    /// Cross product of two `Vec3` vectors.
    ///
    /// # Examples
    /// ```
    /// # use engine::math::vecs::vec3::Vec3;
    /// let a = Vec3::new(1.0, 0.0, 0.0);
    /// let b = Vec3::new(0.0, 1.0, 0.0);
    /// assert_eq!(a.cross(b), Vec3::new(0.0, 0.0, 1.0));
    /// ```
    #[inline]
    pub fn cross(self, other: Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }
}

impl fmt::Display for Vec3 {
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
        write!(f, "Vec3 ({}, {}, {})", x, y, z)
    }
}

// --- Arithmetic Implementation ---

impl Add for Vec3 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl Sub for Vec3 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl Mul for Vec3 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x * rhs.x,
            y: self.y * rhs.y,
            z: self.z * rhs.z,
        }
    }
}

impl Mul<f64> for Vec3 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: f64) -> Self::Output {
        Self {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}

impl Div for Vec3 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x / rhs.x,
            y: self.y / rhs.y,
            z: self.z / rhs.z,
        }
    }
}

impl Div<f64> for Vec3 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: f64) -> Self::Output {
        Self {
            x: self.x / rhs,
            y: self.y / rhs,
            z: self.z / rhs,
        }
    }
}

impl Neg for Vec3 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self::Output {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

impl AddAssign for Vec3 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
    }
}

impl SubAssign for Vec3 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.x -= rhs.x;
        self.y -= rhs.y;
        self.z -= rhs.z;
    }
}

impl MulAssign for Vec3 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        self.x *= rhs.x;
        self.y *= rhs.y;
        self.z *= rhs.z;
    }
}

impl MulAssign<f64> for Vec3 {
    #[inline]
    fn mul_assign(&mut self, rhs: f64) {
        self.x *= rhs;
        self.y *= rhs;
        self.z *= rhs;
    }
}

impl DivAssign for Vec3 {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        self.x /= rhs.x;
        self.y /= rhs.y;
        self.z /= rhs.z;
    }
}

impl DivAssign<f64> for Vec3 {
    #[inline]
    fn div_assign(&mut self, rhs: f64) {
        self.x /= rhs;
        self.y /= rhs;
        self.z /= rhs;
    }
}

// --- VecOperations Implementation ---

impl VecOperations for Vec3 {
    /// Calculates the dot product of two [Vec3].
    ///
    /// # Examples
    /// ```
    /// # use engine::math::vecs::vec3::Vec3;
    /// # use engine::math::vecs::VecOperations;
    /// let a = Vec3::new(2.0, 2.0, 2.0);
    /// let b = Vec3::new(2.0, 2.0, 2.0);
    /// assert_eq!(a.dot(b), 12.0);
    /// ```
    #[inline]
    fn dot(self, other: Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Calculates square length of [Vec3]. (Faster than [length])
    ///
    /// # Examples
    /// ```
    /// # use engine::math::vecs::vec3::Vec3;
    /// # use engine::math::vecs::VecOperations;
    /// let a = Vec3::new(2.0, 2.0, 2.0);
    /// assert_eq!(a.length2(), 12.0);
    /// ```
    #[inline]
    fn length2(self) -> f64 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    /// Calculates sqrt length of [Vec3].
    ///
    /// # Examples
    /// ```
    /// # use engine::math::vecs::vec3::Vec3;
    /// # use engine::math::vecs::VecOperations;
    /// let a = Vec3::new(2.0, 2.0, 2.0);
    /// assert_eq!(a.length(), 12.0f64.sqrt());
    /// ```
    #[inline]
    fn length(self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    /// Normalize [Vec3]. Resulting length will be 1.0.
    ///
    /// # Examples
    /// ```
    /// # use engine::math::vecs::vec3::Vec3;
    /// # use engine::math::vecs::VecOperations;
    /// let a = Vec3::new(2.0, 2.0, 2.0).norm();
    /// assert!((a.length() - 1.0).abs() < f64::EPSILON);
    /// ```
    #[inline]
    fn norm(self) -> Self {
        self / self.length()
    }

    /// Calculates the Euclidean distance between two [Vec3].
    ///
    /// # Examples
    /// ```
    /// # use engine::math::vecs::vec3::Vec3;
    /// # use engine::math::vecs::VecOperations;
    /// let a = Vec3::new(2.0, 2.0, 2.0);
    /// let b = Vec3::new(2.0, 2.0, 2.0);
    /// assert_eq!(a.distance(b), 0.0);
    /// ```
    #[inline]
    fn distance(self, other: Self) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2) + (self.z - other.z).powi(2))
            .sqrt()
    }

    /// Linear interpolation between two [Vec3] by a factor `t`.
    ///
    /// # Examples
    /// ```
    /// # use engine::math::vecs::vec3::Vec3;
    /// # use engine::math::vecs::VecOperations;
    /// let a = Vec3::new(0.0, 0.0, 0.0);
    /// let b = Vec3::new(2.0, 2.0, 2.0);
    /// assert_eq!(a.lerp(b, 0.5), Vec3::new(1.0, 1.0, 1.0));
    /// ```
    #[inline]
    fn lerp(self, other: Self, t: f64) -> Self {
        self * (1.0 - t) + other * t
    }

    /// Returns a [Vec3] containing the minimum components of two vectors.
    ///
    /// # Examples
    /// ```
    /// # use engine::math::vecs::vec3::Vec3;
    /// # use engine::math::vecs::VecOperations;
    /// let a = Vec3::new(0.0, 5.0, -1.0);
    /// let b = Vec3::new(1.0, 2.0, 2.0);
    /// assert_eq!(a.min(b), Vec3::new(0.0, 2.0, -1.0));
    /// ```
    #[inline]
    fn min(self, other: Self) -> Self {
        Self {
            x: self.x.min(other.x),
            y: self.y.min(other.y),
            z: self.z.min(other.z),
        }
    }

    /// Returns a [Vec3] containing the maximum components of two vectors.
    ///
    /// # Examples
    /// ```
    /// # use engine::math::vecs::vec3::Vec3;
    /// # use engine::math::vecs::VecOperations;
    /// let a = Vec3::new(0.0, 5.0, -1.0);
    /// let b = Vec3::new(1.0, 2.0, 2.0);
    /// assert_eq!(a.max(b), Vec3::new(1.0, 5.0, 2.0));
    /// ```
    #[inline]
    fn max(self, other: Self) -> Self {
        Self {
            x: self.x.max(other.x),
            y: self.y.max(other.y),
            z: self.z.max(other.z),
        }
    }

    /// Clamps the [Vec3] components between `min` and `max` vectors.
    ///
    /// # Examples
    /// ```
    /// # use engine::math::vecs::vec3::Vec3;
    /// # use engine::math::vecs::VecOperations;
    /// let min = Vec3::new(-1.0, -1.0, -1.0);
    /// let max = Vec3::new(1.0, 1.0, 1.0);
    /// let v = Vec3::new(2.0, -2.0, 0.0);
    /// assert_eq!(v.clamp(min, max), Vec3::new(1.0, -1.0, 0.0));
    /// ```
    #[inline]
    fn clamp(self, min: Self, max: Self) -> Self {
        Self {
            x: self.x.clamp(min.x, max.x),
            y: self.y.clamp(min.y, max.y),
            z: self.z.clamp(min.z, max.z),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_vec3_display() {
        let a = Vec3 {
            x: 2.0,
            y: 3.0,
            z: 4.0,
        };
        assert_eq!(format!("{}", a), "Vec3 (2, 3, 4)");

        let b = Vec3 {
            x: 1.5,
            y: 0.25,
            z: 0.125,
        };
        assert_eq!(format!("{}", b), "Vec3 (1.5, 0.25, 0.125)");

        let c = Vec3 {
            x: 1.1234567890123,
            y: 0.0000000001,
            z: 3.141592653589,
        };
        assert_eq!(
            format!("{}", c),
            "Vec3 (1.123456789, 0.0000000001, 3.1415926536)"
        );

        let d = Vec3 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        };
        assert_eq!(format!("{}", d), "Vec3 (0, 0, 0)");
    }

    #[test]
    fn test_vec3_add() {
        let a = Vec3 {
            x: 2.0,
            y: 2.0,
            z: 2.0,
        };
        let b = Vec3 {
            x: 2.0,
            y: 2.0,
            z: 2.0,
        };
        assert_eq!(
            a + b,
            Vec3 {
                x: 4.0,
                y: 4.0,
                z: 4.0
            }
        )
    }

    #[test]
    fn test_vec3_sub() {
        let a = Vec3 {
            x: 2.0,
            y: 2.0,
            z: 2.0,
        };
        let b = Vec3 {
            x: 2.0,
            y: 2.0,
            z: 2.0,
        };
        assert_eq!(
            a - b,
            Vec3 {
                x: 0.0,
                y: 0.0,
                z: 0.0
            }
        )
    }

    #[test]
    fn test_vec3_mul() {
        let a = Vec3 {
            x: 2.0,
            y: 2.0,
            z: 2.0,
        };
        let b = Vec3 {
            x: 2.0,
            y: 2.0,
            z: 2.0,
        };
        assert_eq!(
            a * b,
            Vec3 {
                x: 4.0,
                y: 4.0,
                z: 4.0
            }
        )
    }

    #[test]
    fn test_vec3_mulscalar() {
        let a = Vec3 {
            x: 2.0,
            y: 2.0,
            z: 2.0,
        };
        assert_eq!(
            a * 2.0,
            Vec3 {
                x: 4.0,
                y: 4.0,
                z: 4.0
            }
        )
    }

    #[test]
    fn test_vec3_cross() {
        let a = Vec3 {
            x: 1.0,
            y: 0.0,
            z: 0.0,
        };
        let b = Vec3 {
            x: 0.0,
            y: 1.0,
            z: 0.0,
        };
        assert_eq!(
            a.cross(b),
            Vec3 {
                x: 0.0,
                y: 0.0,
                z: 1.0
            }
        );
        assert_eq!(
            b.cross(a),
            Vec3 {
                x: 0.0,
                y: 0.0,
                z: -1.0
            }
        );
    }

    #[test]
    fn test_vec3_div() {
        let a = Vec3 {
            x: 2.0,
            y: 2.0,
            z: 2.0,
        };
        let b = Vec3 {
            x: 2.0,
            y: 2.0,
            z: 2.0,
        };
        assert_eq!(
            a / b,
            Vec3 {
                x: 1.0,
                y: 1.0,
                z: 1.0
            }
        )
    }

    #[test]
    fn test_vec3_divscalar() {
        let a = Vec3 {
            x: 2.0,
            y: 2.0,
            z: 2.0,
        };
        assert_eq!(
            a / 2.0,
            Vec3 {
                x: 1.0,
                y: 1.0,
                z: 1.0
            }
        )
    }

    #[test]
    fn test_vec3_neg() {
        let a = Vec3 {
            x: 2.0,
            y: 2.0,
            z: 2.0,
        };
        assert_eq!(
            -a,
            Vec3 {
                x: -2.0,
                y: -2.0,
                z: -2.0
            }
        )
    }

    #[test]
    fn test_vec3_addasign() {
        let mut a = Vec3 {
            x: 2.0,
            y: 2.0,
            z: 2.0,
        };
        a += Vec3 {
            x: 2.0,
            y: 2.0,
            z: 2.0,
        };
        assert_eq!(
            a,
            Vec3 {
                x: 4.0,
                y: 4.0,
                z: 4.0
            }
        )
    }

    #[test]
    fn test_vec3_subassign() {
        let mut a = Vec3 {
            x: 2.0,
            y: 2.0,
            z: 2.0,
        };
        a -= Vec3 {
            x: 2.0,
            y: 2.0,
            z: 2.0,
        };
        assert_eq!(
            a,
            Vec3 {
                x: 0.0,
                y: 0.0,
                z: 0.0
            }
        )
    }

    #[test]
    fn test_vec3_mulassign() {
        let mut a = Vec3 {
            x: 2.0,
            y: 2.0,
            z: 2.0,
        };
        a *= Vec3 {
            x: 2.0,
            y: 2.0,
            z: 2.0,
        };
        assert_eq!(
            a,
            Vec3 {
                x: 4.0,
                y: 4.0,
                z: 4.0
            }
        )
    }

    #[test]
    fn test_vec3_mulassign_scalar() {
        let mut a = Vec3 {
            x: 2.0,
            y: 2.0,
            z: 2.0,
        };
        a *= 2.0;
        assert_eq!(
            a,
            Vec3 {
                x: 4.0,
                y: 4.0,
                z: 4.0
            }
        )
    }

    #[test]
    fn test_vec3_divassign() {
        let mut a = Vec3 {
            x: 2.0,
            y: 2.0,
            z: 2.0,
        };
        a /= Vec3 {
            x: 2.0,
            y: 2.0,
            z: 2.0,
        };
        assert_eq!(
            a,
            Vec3 {
                x: 1.0,
                y: 1.0,
                z: 1.0
            }
        )
    }

    #[test]
    fn test_vec3_divassign_scalar() {
        let mut a = Vec3 {
            x: 2.0,
            y: 2.0,
            z: 2.0,
        };
        a /= 2.0;
        assert_eq!(
            a,
            Vec3 {
                x: 1.0,
                y: 1.0,
                z: 1.0
            }
        )
    }

    #[test]
    fn test_vec3_dot() {
        let a = Vec3 {
            x: 2.0,
            y: 2.0,
            z: 2.0,
        };
        let b = Vec3 {
            x: 2.0,
            y: 2.0,
            z: 2.0,
        };
        assert_eq!(a.dot(b), 12.0)
    }

    #[test]
    fn test_vec3_length2() {
        let a = Vec3 {
            x: 2.0,
            y: 2.0,
            z: 2.0,
        };
        assert_eq!(a.length2(), 12.0)
    }

    #[test]
    fn test_vec3_length() {
        let a = Vec3 {
            x: 2.0,
            y: 2.0,
            z: 2.0,
        };
        assert_eq!(a.length(), 12.0f64.sqrt())
    }

    #[test]
    fn test_vec3_norm() {
        let a = Vec3 {
            x: 2.0,
            y: 2.0,
            z: 2.0,
        }
        .norm();
        assert!((a.length() - 1.0).abs() < f64::EPSILON)
    }

    #[test]
    fn test_vec3_distance() {
        let a = Vec3 {
            x: 2.0,
            y: 2.0,
            z: 2.0,
        };
        let b = Vec3 {
            x: 2.0,
            y: 2.0,
            z: 2.0,
        };
        assert_eq!(a.distance(b), 0.0)
    }

    #[test]
    fn test_vec3_lerp() {
        let a = Vec3 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        };
        let b = Vec3 {
            x: 2.0,
            y: 2.0,
            z: 2.0,
        };
        assert_eq!(
            a.lerp(b, 0.5),
            Vec3 {
                x: 1.0,
                y: 1.0,
                z: 1.0
            }
        )
    }

    #[test]
    fn test_vec3_min() {
        let a = Vec3 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        };
        let b = Vec3 {
            x: 1.0,
            y: 1.0,
            z: 1.0,
        };
        assert_eq!(
            a.min(b),
            Vec3 {
                x: 0.0,
                y: 0.0,
                z: 0.0
            }
        )
    }

    #[test]
    fn test_vec3_max() {
        let a = Vec3 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        };
        let b = Vec3 {
            x: 1.0,
            y: 1.0,
            z: 1.0,
        };
        assert_eq!(
            a.max(b),
            Vec3 {
                x: 1.0,
                y: 1.0,
                z: 1.0
            }
        )
    }

    #[test]
    fn test_vec3_clamp() {
        let max = Vec3 {
            x: 1.0,
            y: 1.0,
            z: 1.0,
        };
        let min = Vec3 {
            x: -1.0,
            y: -1.0,
            z: -1.0,
        };

        let out_min = Vec3 {
            x: -2.0,
            y: -2.0,
            z: -2.0,
        };
        let mid = Vec3 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        };
        let out_max = Vec3 {
            x: 2.0,
            y: 2.0,
            z: 2.0,
        };

        let mixed_out_min = Vec3 {
            x: -2.0,
            y: 0.0,
            z: 2.0,
        };
        let mixed_out_max = Vec3 {
            x: 2.0,
            y: 0.0,
            z: -2.0,
        };

        assert_eq!(
            out_min.clamp(min, max),
            Vec3 {
                x: -1.0,
                y: -1.0,
                z: -1.0
            }
        );
        assert_eq!(
            mid.clamp(min, max),
            Vec3 {
                x: 0.0,
                y: 0.0,
                z: 0.0
            }
        );
        assert_eq!(
            out_max.clamp(min, max),
            Vec3 {
                x: 1.0,
                y: 1.0,
                z: 1.0
            }
        );

        assert_eq!(
            mixed_out_min.clamp(min, max),
            Vec3 {
                x: -1.0,
                y: 0.0,
                z: 1.0
            }
        );
        assert_eq!(
            mixed_out_max.clamp(min, max),
            Vec3 {
                x: 1.0,
                y: 0.0,
                z: -1.0
            }
        );
    }
}
