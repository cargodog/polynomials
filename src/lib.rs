#![cfg_attr(not(feature = "std"), no_std)]

use core::cmp::PartialEq;
use core::convert::From;
use core::fmt::Display;
use core::ops::Neg;
use core::ops::{Add, AddAssign};
use core::ops::{Div, DivAssign};
use core::ops::{Index, IndexMut};
use core::ops::{Mul, MulAssign};
use core::ops::{Sub, SubAssign};
use core::slice::SliceIndex;

use serde::{Serialize, Deserialize};

#[macro_use]
pub mod sparse;

#[cfg_attr(test, macro_use)]
extern crate alloc;
use alloc::vec::{IntoIter, Vec};

/// A [`Polynomial`] is just a vector of coefficients. Each coefficient corresponds to a power of
/// `x` in increasing order. For example, the following polynomial is equal to 4x^2 + 3x - 9.
///
/// ```
/// # #[macro_use] extern crate polynomials;
/// # fn main() {
/// // Construct polynomial 4x^2 + 3x - 9
/// let mut a = poly![-9, 3, 4];
/// assert_eq!(a[0], -9);
/// assert_eq!(a[1], 3);
/// assert_eq!(a[2], 4);
/// # }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Polynomial<T>(Vec<T>);

impl<T> Polynomial<T> {
    /// Create a new, empty, instance of a polynomial.
    pub fn new() -> Polynomial<T> {
        Polynomial(Vec::<T>::new())
    }

    /// Adds a new coefficient to the [`Polynomial`], in the next highest order position.
    ///
    /// ```
    /// # #[macro_use] extern crate polynomials;
    /// # fn main() {
    /// let mut a = poly![-8, 2, 4];
    /// a.push(7);
    /// assert_eq!(a, poly![-8, 2, 4, 7]);
    /// # }
    /// ```
    pub fn push(&mut self, value: T) {
        self.0.push(value);
    }

    /// Removes the highest order coefficient from the [`Polynomial`].
    ///
    /// ```
    /// # #[macro_use] extern crate polynomials;
    /// # fn main() {
    /// let mut a = poly![-8, 2, 4];
    /// assert_eq!(a.pop().unwrap(), 4);
    /// assert_eq!(a, poly![-8, 2]);
    /// # }
    /// ```
    pub fn pop(&mut self) -> Option<T> {
        self.0.pop()
    }

    /// Calculates the degree of a [`Polynomial`].
    ///
    /// The following polynomial is of degree 2: (4x^2 + 2x - 8)
    ///
    /// ```
    /// # #[macro_use] extern crate polynomials;
    /// # fn main() {
    /// let a = poly![-8, 2, 4];
    /// assert_eq!(a.degree(), 2);
    /// # }
    /// ```
    pub fn degree(&self) -> usize
    where
        T: Sub<T, Output = T> + Eq + Copy,
    {
        let mut deg = self.0.len();
        for _ in 0..self.0.len() {
            deg -= 1;

            // Generic test if non-zero
            if self[deg] != self[deg] - self[deg] {
                break;
            }
        }
        deg
    }

    /// Evaluate a [`Polynomial`] for some value `x`.
    ///
    /// The following example evaluates the polynomial (4x^2 + 2x - 8) for x = 3.
    ///
    /// ```
    /// # #[macro_use] extern crate polynomials;
    /// # fn main() {
    /// let a = poly![-8, 2, 4];
    /// assert_eq!(a.eval(3).unwrap(), 34);
    /// # }
    /// ```
    pub fn eval<X>(&self, x: X) -> Option<T>
    where
        T: AddAssign + Copy,
        X: MulAssign + Mul<T, Output = T> + Copy,
    {
        if self.0.len() == 0 {
            None
        } else {
            let mut p = x; // running power of `x`
            let mut res = self[0];
            for i in 1..self.0.len() {
                res += p * self[i];
                p *= x;
            }
            Some(res)
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.0.iter()
    }

    pub fn into_iter(self) -> impl IntoIterator<Item = T, IntoIter = IntoIter<T>> {
        self.0.into_iter()
    }
}

impl<T> From<Vec<T>> for Polynomial<T> {
    fn from(v: Vec<T>) -> Self {
        Polynomial(v)
    }
}

impl<T> Into<Vec<T>> for Polynomial<T> {
    fn into(self) -> Vec<T> {
        self.0
    }
}

impl<T, I: SliceIndex<[T]>> Index<I> for Polynomial<T> {
    type Output = I::Output;

    fn index(&self, index: I) -> &Self::Output {
        &self.0[index]
    }
}

impl<T, I: SliceIndex<[T]>> IndexMut<I> for Polynomial<T> {
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        &mut self.0[index]
    }
}

/// Add two [`Polynomial`]s.
///
/// The following example adds two polynomials:
/// (4x^2 + 2x - 8) + (x + 1) = (4x^2 + 3x - 7)
///
/// ```
/// # #[macro_use] extern crate polynomials;
/// # fn main() {
/// let a = poly![-8, 2, 4];
/// let b = poly![1, 1];
/// assert_eq!(a + b, poly![-7, 3, 4]);
/// # }
/// ```
impl<T: Add<Output = T>> Add for Polynomial<T>
where
    T: Add + Copy + Clone,
{
    type Output = Self;

    fn add(mut self, other: Self) -> Self::Output {
        self += other;
        self
    }
}

impl<T> AddAssign for Polynomial<T>
where
    T: Add<Output = T> + Copy,
{
    fn add_assign(&mut self, rhs: Self) {
        let min_len = if self.0.len() < rhs.0.len() {
            self.0.len()
        } else {
            rhs.0.len()
        };
        if self.0.len() == min_len {
            for i in min_len..rhs.0.len() {
                self.push(rhs[i]);
            }
        }
        for i in 0..min_len {
            self[i] = self[i] + rhs[i];
        }
    }
}

/// Subtract two [`Polynomial`]s.
///
/// The following example subtracts two polynomials:
/// (4x^2 + 2x - 8) - (x + 1) = (4x^2 + x - 9)
///
/// ```
/// # #[macro_use] extern crate polynomials;
/// # fn main() {
/// let a = poly![-8, 2, 4];
/// let b = poly![1, 1];
/// assert_eq!(a - b, poly![-9, 1, 4]);
/// # }
/// ```
impl<T: Sub<Output = T>> Sub for Polynomial<T>
where
    T: Sub + Neg<Output = T> + Copy + Clone,
{
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        let mut diff = self.clone();
        diff -= other;
        diff
    }
}

impl<T> SubAssign for Polynomial<T>
where
    T: Sub<Output = T> + Neg<Output = T> + Copy,
{
    fn sub_assign(&mut self, rhs: Self) {
        let min_len = if self.0.len() < rhs.0.len() {
            self.0.len()
        } else {
            rhs.0.len()
        };
        if self.0.len() == min_len {
            for i in min_len..rhs.0.len() {
                self.push(-rhs[i]);
            }
        }
        for i in 0..min_len {
            self[i] = self[i] - rhs[i];
        }
    }
}

/// Multiply two [`Polynomial`]s.
///
/// The following example multiplies two polynomials:
/// (4x^2 + 2x - 8) * (x + 1) = (4x^3 + 6x^2 - 6x - 8)
///
/// ```
/// # #[macro_use] extern crate polynomials;
/// # fn main() {
/// let a = poly![-8, 2, 4];
/// let b = poly![1, 1];
/// assert_eq!(a * b, poly![-8, -6, 6, 4]);
/// # }
/// ```
impl<T> Mul<T> for Polynomial<T>
where
    T: MulAssign + Copy,
{
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        let mut prod = self.clone();
        prod *= rhs;
        prod
    }
}

impl<T> MulAssign<T> for Polynomial<T>
where
    T: MulAssign + Copy,
{
    fn mul_assign(&mut self, rhs: T) {
        for i in 0..self.0.len() {
            self[i] *= rhs;
        }
    }
}

/// Multiply a [`Polynomial`] by some value.
///
/// The following example multiplies a polynomial (4x^2 + 2x - 8) by 2:
///
/// ```
/// # #[macro_use] extern crate polynomials;
/// # fn main() {
/// let p = poly![-8, 2, 4] * 2;
/// assert_eq!(p, poly![-16, 4, 8]);
/// # }
/// ```
impl<T> Mul for Polynomial<T>
where
    T: Mul<Output = T> + AddAssign + Sub<Output = T>,
    T: Copy + Clone,
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut new = self.clone();
        new *= rhs;
        new
    }
}

impl<T> MulAssign for Polynomial<T>
where
    T: Mul<Output = T> + AddAssign + Sub<Output = T>,
    T: Copy + Clone,
{
    fn mul_assign(&mut self, rhs: Self) {
        let orig = self.clone();

        // One of the vectors must be non-empty
        if self.0.len() > 0 || rhs.0.len() > 0 {
            // Since core::num does not provide the `Zero()` trait
            // this hack lets us calculate zero from any generic
            let zero = if self.0.len() > 0 {
                self[0] - self[0]
            } else {
                rhs[0] - rhs[0]
            };

            // Clear `self`
            for i in 0..self.0.len() {
                self.0[i] = zero;
            }

            // Resize vector with size M + N - 1
            self.0.resize(self.0.len() + rhs.0.len() - 1, zero);

            // Calculate product
            for i in 0..orig.0.len() {
                for j in 0..rhs.0.len() {
                    self[i + j] += orig[i] * rhs[j];
                }
            }
        }
    }
}

/// Divide a [`Polynomial`] by some value.
///
/// The following example divides a polynomial (4x^2 + 2x - 8) by 2:
///
/// ```
/// # #[macro_use] extern crate polynomials;
/// # fn main() {
/// let p = poly![-8, 2, 4] / 2;
/// assert_eq!(p, poly![-4, 1, 2]);
/// # }
/// ```
impl<T> Div<T> for Polynomial<T>
where
    T: DivAssign + Copy,
{
    type Output = Self;

    fn div(self, rhs: T) -> Self::Output {
        let mut prod = self.clone();
        prod /= rhs;
        prod
    }
}

impl<T> DivAssign<T> for Polynomial<T>
where
    T: DivAssign + Copy,
{
    fn div_assign(&mut self, rhs: T) {
        for i in 0..self.0.len() {
            self[i] /= rhs;
        }
    }
}

impl<T> PartialEq for Polynomial<T>
where
    T: Sub<T, Output = T> + Eq + Copy,
{
    fn eq(&self, other: &Self) -> bool {
        let degree = self.degree();
        if degree != other.degree() {
            return false;
        }
        for i in 0..=degree {
            if self[i] != other[i] {
                return false;
            }
        }
        true
    }
}

impl<T> Eq for Polynomial<T> where T: Sub<T, Output = T> + Eq + Copy {}
impl<T> Display for Polynomial<T> where T: Display+Copy+Sub<T, Output = T> + Eq+num::Num+num::Signed{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let degree=self.degree();
        let mut formatted_equation=String::new();
        for (i,coefficient) in self.0.iter().enumerate(){
            if coefficient.is_zero(){
                continue;
            }
            if !coefficient.is_one()||degree-i==0{
                formatted_equation+=&coefficient.to_string();
            }
            if degree-i!=0{
                formatted_equation+=&format!("x^{}",degree-i);
            }
            if i<degree&&coefficient.is_positive(){
                formatted_equation+="+"
            }
        }
        write!(f,"{}",formatted_equation)
    }
}
/// Creates a [`Polynomial`] from a list of coefficients in ascending order.
///
/// This is a wrapper around the `vec!` macro, to instantiate a polynomial from
/// a vector of coefficients.
///
/// `poly!` allows `Polynomial`s to be defined with the same syntax as array expressions.
/// There are two forms of this macro:
///
/// - Create a [`Polynomial`] containing a given list of coefficients:
///
/// ```
/// # #[macro_use] extern crate polynomials;
/// # fn main() {
/// let p = poly![1, 2, 3]; // 3x^2 + 2x + 1
/// assert_eq!(p[0], 1);
/// assert_eq!(p[1], 2);
/// assert_eq!(p[2], 3);
/// # }
/// ```
///
/// - Create a [`Polynomial`] from a given coefficient and size:
///
/// ```
/// # #[macro_use] extern crate polynomials;
/// # fn main() {
/// let p = poly![1; 3]; // x^2 + x + 1
/// assert_eq!(p, poly![1, 1, 1]);
/// # }
/// ```
#[macro_export]
macro_rules! poly {
    ($($args:tt)*) => (
         $crate::Polynomial::from(vec![$($args)*])
     );
}

#[cfg(test)]
mod tests {
    use super::*;
    use sparse::SparsePolynomial;

    #[test]
    fn degree() {
        assert_eq!(poly![8, 6, 2, 3].degree(), 3);
        assert_eq!(poly![8, 6, 2, 3].degree(), 3);
        assert_eq!(poly![0, 0, 6, 2, 3].degree(), 4);
        assert_eq!(poly![0, 0].degree(), 0);
        assert_eq!(poly![0, 99].degree(), 1);
        assert_eq!(poly![99, 0].degree(), 0);
    }

    #[test]
    fn eval() {
        assert_eq!(poly![1, 1, 1, 1].eval(1).unwrap(), 4);
        assert_eq!(poly![-2, -2, -2, -2].eval(1).unwrap(), -8);
        assert_eq!(poly![100, 0, 0, 0].eval(9).unwrap(), 100);
        assert_eq!(poly![0, 1, 0, 0].eval(9).unwrap(), 9);
        assert_eq!(poly![0, 0, -1, 0].eval(9).unwrap(), -81);
        assert_eq!(poly![0, -9, 0, 40].eval(2).unwrap(), 302);
    }

    #[test]
    fn iter() {
        assert_eq!(poly![0, -9, 0, 40].iter().sum::<isize>(), 31);
    }

    #[test]
    fn add() {
        let a = poly![-200, 6, 2, 3, 53, 0, 0]; // Higher order 0s should be ignored
        let b = poly![-1, -6, -7, 0, 1000];
        let c = poly![-201, 0, -5, 3, 1053];
        assert_eq!(a.clone() + b.clone(), c);
        assert_eq!(b + a, c);
    }

    #[test]
    fn add_assign() {
        let mut a = poly![-200, 6, 2, 3, 53, 0, 0]; // Higher order 0s should be ignored
        let b = poly![-1, -6, -7, 0, 1000];
        let c = poly![-201, 0, -5, 3, 1053];
        a += b;
        assert_eq!(a, c);

        let mut a = poly![1]; // Low degree should be expanded
        let b = poly![0, 1];
        let c = poly![1, 1];
        a += b;
        assert_eq!(a, c);
    }

    #[test]
    fn sub() {
        let a = poly![-200, 6, 2, 3, 53, 0, 0]; // Higher order 0s should be ignored
        let b = poly![-1, -6, -7, 0, 1000];
        let c = poly![-199, 12, 9, 3, -947];
        let d = poly![199, -12, -9, -3, 947];
        assert_eq!(a.clone() - b.clone(), c);
        assert_eq!(b - a, d);
    }

    #[test]
    fn sub_assign() {
        let mut a = poly![-200, 6, 2, 3, 53, 0, 0]; // Higher order 0s should be ignored
        let b = poly![-1, -6, -7, 0, 1000];
        let c = poly![-199, 12, 9, 3, -947];
        a -= b;
        assert_eq!(a, c);

        let mut a = poly![1]; // Low degree should be expanded
        let b = poly![0, 1];
        let c = poly![1, -1];
        a -= b;
        assert_eq!(a, c);
    }

    #[test]
    fn mul() {
        let a = poly![1, 0, 0]; // Higher order 0s should be ignored
        let b = poly![0];
        let c = poly![0];
        assert_eq!(a * b, c);

        let a = poly![-7];
        let b = poly![4];
        let c = poly![-28];
        assert_eq!(a * b, c);

        let a = poly![0, 1];
        let b = poly![4];
        let c = poly![0, 4];
        assert_eq!(a * b, c);

        let a = poly![0, -1];
        let b = poly![0, 1];
        let c = poly![0, 0, -1];
        assert_eq!(a * b, c);
    }

    #[test]
    fn mul_assign() {
        let mut a = poly![1, 0, 0]; // Higher order 0s should be ignored
        let b = poly![0];
        let c = poly![0];
        a *= b;
        assert_eq!(a, c);

        let mut a = poly![-7];
        let b = poly![4];
        let c = poly![-28];
        a *= b;
        assert_eq!(a, c);

        let mut a = poly![0, 1];
        let b = poly![4];
        let c = poly![0, 4];
        a *= b;
        assert_eq!(a, c);

        let mut a = poly![0, -1];
        let b = poly![0, 1];
        let c = poly![0, 0, -1];
        a *= b;
        assert_eq!(a, c);
    }

    #[test]
    fn mul_by_value() {
        let a = poly![1, 2, 3];
        let b = poly![2, 4, 6];
        assert_eq!(a * 2, b);

        let mut a = poly![1, 2, 3];
        let b = poly![2, 4, 6];
        a *= 2;
        assert_eq!(a, b);
    }

    #[test]
    fn div_by_value() {
        let a = poly![2, 4, 6];
        let b = poly![1, 2, 3];
        assert_eq!(a / 2, b);

        let mut a = poly![2, 4, 6];
        let b = poly![1, 2, 3];
        a /= 2;
        assert_eq!(a, b);
    }

    #[test]
    fn equality() {
        let a = poly![1, 0];
        let b = poly![-1, 0];
        assert!(a != b);
    }
    #[test]
    fn polynomial_display(){
        let mut a=poly![1,4,5];
        assert_eq!(format!("{}",a),"x^2+4x^1+5");
        a=poly![1,0,5];
        assert_eq!(format!("{}",a),"x^2+5");
    }
    #[test]
    fn sparse_degree() {
        assert_eq!(SparsePolynomial::from(vec![(0,8), (1,6), (100,2), (5,3)]).degree(), 100);
        assert_eq!(SparsePolynomial::from(vec![(0,8), (5,3)]).degree(), 5);
        assert_eq!(SparsePolynomial::from(vec![(0,8)]).degree(), 0);
    }

    #[test]
    fn sparse_add() {
        let mut a = SparsePolynomial::from(vec![(0,1),(1,1)]);
        let mut b = SparsePolynomial::from(vec![(0,1),(1,1)]);
        let mut c = a + b;
        assert_eq!(SparsePolynomial::from(vec![(0,2), (1,2)]), c);

        a = SparsePolynomial::from(vec![(0,-1),(1,1)]);
        b = SparsePolynomial::from(vec![(0,1),(1,1)]);
        c = a + b;
        assert_eq!(SparsePolynomial::from(vec![(1,2)]), c);
    }

    #[test]
    fn sparse_mul() {
        let a = SparsePolynomial::from(vec![(0,1),(1,1)]);
        let b = SparsePolynomial::from(vec![(0,1),(1,1)]);
        let c = a * b;
        assert_eq!(SparsePolynomial::from(vec![(0,1),(1,2),(2,1)]), c);
    }

    #[test]
    fn sparse_mul_high_degree() {
        let a = SparsePolynomial::from(vec![(0,-1),(12,1)]);
        let b = SparsePolynomial::from(vec![(12,9),(15,1),(100,3)]);
        let c = a * b;
        // checked on wolfram alpha
        assert_eq!(SparsePolynomial::from(vec![(12,-9),(15,-1),(24,9),(27,1),(100,-3),(112,3)]), c);
    }

    #[test]
    fn sparse_eval() {
        let a = SparsePolynomial::from(vec![(0,-1),(12,1)]);
        // checked on wolfram alpha
        let mut y = a.eval(1);
        assert_eq!(y, 0.into());
        y = a.eval(2);
        assert_eq!(y, 4_095.into());
        y = a.eval(3);
        assert_eq!(y, 531_440.into());
        y = a.eval(4);
        assert_eq!(y, 16_777_215.into());
        y = a.eval(5);
        assert_eq!(y, 244_140_624.into());
    }    
}
