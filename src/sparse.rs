use core::cmp::PartialEq;
use core::convert::From;
use core::ops::Neg;
use core::ops::{Add, AddAssign};
use core::ops::{Div, DivAssign};
use core::ops::{Index, IndexMut};
use core::ops::{Mul, MulAssign};
use core::ops::{Sub, SubAssign};
use core::slice::SliceIndex;
use serde::{Serialize, Deserialize};

use alloc::vec::{IntoIter, Vec};

/// A [`SparsePolynomial`] is just a vector of coefficients. Each coefficient corresponds to a power of
/// `x` in increasing order. For example, the following polynomial is equal to 4x^2 + 3x - 9.
///
/// ```
/// # #[macro_use] extern crate polynomials;
/// # fn main() {
/// // Construct polynomial 4x^100 + 3x - 9
/// let mut a = sparse_poly![(-9, 0), (3, 1), (4, 100)];
/// assert_eq!(a[0], -9);
/// assert_eq!(a[1], 3);
/// assert_eq!(a[2], 4);
/// # }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparsePolynomial<T>(Vec<(T,T)>);

impl<T> SparsePolynomial<T> {
    /// Create a new, empty, instance of a polynomial.
    pub fn new() -> SparsePolynomial<T> {
        SparsePolynomial(Vec::<(T,T)>::new())
    }

    /// Adds a new coefficient to the [`SparsePolynomial`], in the next highest order position.
    ///
    /// ```
    /// # #[macro_use] extern crate polynomials;
    /// # fn main() {
    /// let mut a = sparse_poly![-8, 2, 4];
    /// a.push(7);
    /// assert_eq!(a, sparse_poly![-8, 2, 4, 7]);
    /// # }
    /// ```
    pub fn push(&mut self, value: (T,T)) {
        self.0.push(value);
    }

    /// Removes the highest order coefficient from the [`SparsePolynomial`].
    ///
    /// ```
    /// # #[macro_use] extern crate polynomials;
    /// # fn main() {
    /// let mut a = sparse_poly![-8, 2, 4];
    /// assert_eq!(a.pop().unwrap(), 4);
    /// assert_eq!(a, sparse_poly![-8, 2]);
    /// # }
    /// ```
    pub fn pop(&mut self) -> Option<(T,T)> {
        self.0.pop()
    }

    /// Calculates the degree of a [`SparsePolynomial`].
    ///
    /// The following polynomial is of degree 2: (4x^2 + 2x - 8)
    ///
    /// ```
    /// # #[macro_use] extern crate polynomials;
    /// # fn main() {
    /// let a = sparse_poly![-8, 2, 4];
    /// assert_eq!(a.degree(), 2);
    /// # }
    /// ```
    pub fn degree(&self) -> usize
    where
        T: Sub<T, Output = T> + Eq + Copy,
    {
        let deg = 0;
        for i in 0..self.0.len() {
            if self.0[i].1 != 0 {
                if self.0[i].1 > deg {
                    deg = self.0[i].1;
                }
            }
        }
        deg
    }

    /// Evaluate a [`SparsePolynomial`] for some value `x`.
    ///
    /// The following example evaluates the polynomial (4x^2 + 2x - 8) for x = 3.
    ///
    /// ```
    /// # #[macro_use] extern crate polynomials;
    /// # fn main() {
    /// let a = sparse_poly![(-8, 0), (2, 1), (4, 2)];
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
            let mut max_exp = 0;
            let mut storage = vec![];
            let mut storage_exps = vec![];

            let get_next_val = |diff| {
                // if diff is larger, than we can double
                if diff >= max_exp {
                    storage.push(max_val * max_val);
                    storage_exps.push(max_exp * 2);
                } else {
                    // TODO: Optimise using binary search
                    for i in 0..storage.len() {
                        let elt = storage[storage.len() - 1 - i];
                        let elt_exp = storage[storage.len() - 1 - i];
                        if diff >= elt_exp {
                            storage.push(max_val * elt);
                            storage_exps.push(max_val + elt_exp);
                            break;
                        }
                    }
                }

                return (
                    storage[storage.len() - 1],
                    storage_exps[storage_exps.len() - 1],
                );
            }

            let get_existing_val = |exp| {
                // TODO: Optimise using binary search
                for i in 0..storage_exps.len() {
                    let elt = storage[storage.len() - 1 - i];
                    let elt_exp = storage[storage.len() - 1 - i];
                    if exp > elt_exp {
                        storage.push(max_val * elt);
                        storage_exps.push(max_val + elt_exp);
                        return (
                            storage[storage.len() - 1],
                            storage_exps[storage_exps.len() - 1],
                        );
                    }
                }
            }

            for i in 0..self.0.len() {
                if self.0[i].1 == 0 {
                    res += self[i].0;
                } else {
                    let coeff = self[i].0;
                    let exp = self[i].1;
                    if storage.len() == 0 {
                        storage.push(x);
                        storage_exps.push(1);
                        max_exp = 1;
                        // TODO: cleanup bad impl
                        if max_exp == exp {
                            res += self[i].0 * x;
                        } else {
                            while max_exp < exp {
                                let diff = exp - max_exp;
                                let (_, next_exp) = get_next_val(diff);
                                // FIXME: might be redundant
                                max_exp = next_exp;
                            }
                            // ensure we've reached value ^ exponent
                            assert!(max_exp == exp);
                            res += self[i].0 * storage[storage.len() - 1];
                        }
                    } else {
                        // if max_exp > exp, we need to get the largest value y s.t. x^y < x^exp
                        if max_exp > exp {
                            let (mut next_val, mut next_exp) = get_existing_val(exp);
                            while next_val != exp {
                                (next_val, next_exp) = get_existing_val(exp);
                            }
                            res += self[i].0 * next_val;
                        } else {
                            let diff = exp - max_exp;
                            while max_exp < exp {
                                let diff = exp - max_exp;
                                let (_, next_exp) = get_next_val(diff);
                                // FIXME: might be redundant
                                max_exp = next_exp;
                            }
                            // ensure we've reached value ^ exponent
                            assert!(max_exp == exp);
                            res += self[i].0 * storage[storage.len() - 1];
                        }

                        if self[i].1 > max_exp {
                            max_exp = self[i].1;
                        }
                    }
                }
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

impl<T> From<Vec<T>> for SparsePolynomial<T> {
    fn from(v: Vec<T>) -> Self {
        SparsePolynomial(v)
    }
}

impl<T> Into<Vec<T>> for SparsePolynomial<T> {
    fn into(self) -> Vec<T> {
        self.0
    }
}

impl<T, I: SliceIndex<[T]>> Index<I> for SparsePolynomial<T> {
    type Output = I::Output;

    fn index(&self, index: I) -> &Self::Output {
        &self.0[index]
    }
}

impl<T, I: SliceIndex<[T]>> IndexMut<I> for SparsePolynomial<T> {
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        &mut self.0[index]
    }
}

/// Add two [`SparsePolynomial`]s.
///
/// The following example adds two polynomials:
/// (4x^2 + 2x - 8) + (x + 1) = (4x^2 + 3x - 7)
///
/// ```
/// # #[macro_use] extern crate polynomials;
/// # fn main() {
/// let a = sparse_poly![-8, 2, 4];
/// let b = sparse_poly![1, 1];
/// assert_eq!(a + b, sparse_poly![-7, 3, 4]);
/// # }
/// ```
impl<T: Add<Output = T>> Add for SparsePolynomial<T>
where
    T: Add + Copy + Clone,
{
    type Output = Self;

    fn add(mut self, other: Self) -> Self::Output {
        self += other;
        self
    }
}

impl<T> AddAssign for SparsePolynomial<T>
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

/// Subtract two [`SparsePolynomial`]s.
///
/// The following example subtracts two polynomials:
/// (4x^2 + 2x - 8) - (x + 1) = (4x^2 + x - 9)
///
/// ```
/// # #[macro_use] extern crate polynomials;
/// # fn main() {
/// let a = sparse_poly![-8, 2, 4];
/// let b = sparse_poly![1, 1];
/// assert_eq!(a - b, sparse_poly![-9, 1, 4]);
/// # }
/// ```
impl<T: Sub<Output = T>> Sub for SparsePolynomial<T>
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

impl<T> SubAssign for SparsePolynomial<T>
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

/// Multiply two [`SparsePolynomial`]s.
///
/// The following example multiplies two polynomials:
/// (4x^2 + 2x - 8) * (x + 1) = (4x^3 + 6x^2 - 6x - 8)
///
/// ```
/// # #[macro_use] extern crate polynomials;
/// # fn main() {
/// let a = sparse_poly![-8, 2, 4];
/// let b = sparse_poly![1, 1];
/// assert_eq!(a * b, sparse_poly![-8, -6, 6, 4]);
/// # }
/// ```
impl<T> Mul<T> for SparsePolynomial<T>
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

impl<T> MulAssign<T> for SparsePolynomial<T>
where
    T: MulAssign + Copy,
{
    fn mul_assign(&mut self, rhs: T) {
        for i in 0..self.0.len() {
            self[i] *= rhs;
        }
    }
}

/// Multiply a [`SparsePolynomial`] by some value.
///
/// The following example multiplies a polynomial (4x^2 + 2x - 8) by 2:
///
/// ```
/// # #[macro_use] extern crate polynomials;
/// # fn main() {
/// let p = sparse_poly![-8, 2, 4] * 2;
/// assert_eq!(p, sparse_poly![-16, 4, 8]);
/// # }
/// ```
impl<T> Mul for SparsePolynomial<T>
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

impl<T> MulAssign for SparsePolynomial<T>
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

/// Divide a [`SparsePolynomial`] by some value.
///
/// The following example divides a polynomial (4x^2 + 2x - 8) by 2:
///
/// ```
/// # #[macro_use] extern crate polynomials;
/// # fn main() {
/// let p = sparse_poly![-8, 2, 4] / 2;
/// assert_eq!(p, sparse_poly![-4, 1, 2]);
/// # }
/// ```
impl<T> Div<T> for SparsePolynomial<T>
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

impl<T> DivAssign<T> for SparsePolynomial<T>
where
    T: DivAssign + Copy,
{
    fn div_assign(&mut self, rhs: T) {
        for i in 0..self.0.len() {
            self[i] /= rhs;
        }
    }
}

impl<T> PartialEq for SparsePolynomial<T>
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
impl<T> Eq for SparsePolynomial<T> where T: Sub<T, Output = T> + Eq + Copy {}

/// Creates a [`SparsePolynomial`] from a list of coefficients in ascending order.
///
/// This is a wrapper around the `vec!` macro, to instantiate a polynomial from
/// a vector of coefficients.
///
/// `sparse_poly!` allows `SparsePolynomial`s to be defined with the same syntax as array expressions.
/// There are two forms of this macro:
///
/// - Create a [`SparsePolynomial`] containing a given list of coefficients:
///
/// ```
/// # #[macro_use] extern crate polynomials;
/// # fn main() {
/// let p = sparse_poly![1, 2, 3]; // 3x^2 + 2x + 1
/// assert_eq!(p[0], 1);
/// assert_eq!(p[1], 2);
/// assert_eq!(p[2], 3);
/// # }
/// ```
///
/// - Create a [`SparsePolynomial`] from a given coefficient and size:
///
/// ```
/// # #[macro_use] extern crate polynomials;
/// # fn main() {
/// let p = sparse_poly![1; 3]; // x^2 + x + 1
/// assert_eq!(p, sparse_poly![1, 1, 1]);
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
    #[test]
    fn degree() {
        assert_eq!(sparse_poly![8, 6, 2, 3].degree(), 3);
        assert_eq!(sparse_poly![8, 6, 2, 3].degree(), 3);
        assert_eq!(sparse_poly![0, 0, 6, 2, 3].degree(), 4);
        assert_eq!(sparse_poly![0, 0].degree(), 0);
        assert_eq!(sparse_poly![0, 99].degree(), 1);
        assert_eq!(sparse_poly![99, 0].degree(), 0);
    }

    #[test]
    fn eval() {
        assert_eq!(sparse_poly![1, 1, 1, 1].eval(1).unwrap(), 4);
        assert_eq!(sparse_poly![-2, -2, -2, -2].eval(1).unwrap(), -8);
        assert_eq!(sparse_poly![100, 0, 0, 0].eval(9).unwrap(), 100);
        assert_eq!(sparse_poly![0, 1, 0, 0].eval(9).unwrap(), 9);
        assert_eq!(sparse_poly![0, 0, -1, 0].eval(9).unwrap(), -81);
        assert_eq!(sparse_poly![0, -9, 0, 40].eval(2).unwrap(), 302);
    }

    #[test]
    fn iter() {
        assert_eq!(sparse_poly![0, -9, 0, 40].iter().sum::<isize>(), 31);
    }

    #[test]
    fn add() {
        let a = sparse_poly![-200, 6, 2, 3, 53, 0, 0]; // Higher order 0s should be ignored
        let b = sparse_poly![-1, -6, -7, 0, 1000];
        let c = sparse_poly![-201, 0, -5, 3, 1053];
        assert_eq!(a.clone() + b.clone(), c);
        assert_eq!(b + a, c);
    }

    #[test]
    fn add_assign() {
        let mut a = sparse_poly![-200, 6, 2, 3, 53, 0, 0]; // Higher order 0s should be ignored
        let b = sparse_poly![-1, -6, -7, 0, 1000];
        let c = sparse_poly![-201, 0, -5, 3, 1053];
        a += b;
        assert_eq!(a, c);

        let mut a = sparse_poly![1]; // Low degree should be expanded
        let b = sparse_poly![0, 1];
        let c = sparse_poly![1, 1];
        a += b;
        assert_eq!(a, c);
    }

    #[test]
    fn sub() {
        let a = sparse_poly![-200, 6, 2, 3, 53, 0, 0]; // Higher order 0s should be ignored
        let b = sparse_poly![-1, -6, -7, 0, 1000];
        let c = sparse_poly![-199, 12, 9, 3, -947];
        let d = sparse_poly![199, -12, -9, -3, 947];
        assert_eq!(a.clone() - b.clone(), c);
        assert_eq!(b - a, d);
    }

    #[test]
    fn sub_assign() {
        let mut a = sparse_poly![-200, 6, 2, 3, 53, 0, 0]; // Higher order 0s should be ignored
        let b = sparse_poly![-1, -6, -7, 0, 1000];
        let c = sparse_poly![-199, 12, 9, 3, -947];
        a -= b;
        assert_eq!(a, c);

        let mut a = sparse_poly![1]; // Low degree should be expanded
        let b = sparse_poly![0, 1];
        let c = sparse_poly![1, -1];
        a -= b;
        assert_eq!(a, c);
    }

    #[test]
    fn mul() {
        let a = sparse_poly![1, 0, 0]; // Higher order 0s should be ignored
        let b = sparse_poly![0];
        let c = sparse_poly![0];
        assert_eq!(a * b, c);

        let a = sparse_poly![-7];
        let b = sparse_poly![4];
        let c = sparse_poly![-28];
        assert_eq!(a * b, c);

        let a = sparse_poly![0, 1];
        let b = sparse_poly![4];
        let c = sparse_poly![0, 4];
        assert_eq!(a * b, c);

        let a = sparse_poly![0, -1];
        let b = sparse_poly![0, 1];
        let c = sparse_poly![0, 0, -1];
        assert_eq!(a * b, c);
    }

    #[test]
    fn mul_assign() {
        let mut a = sparse_poly![1, 0, 0]; // Higher order 0s should be ignored
        let b = sparse_poly![0];
        let c = sparse_poly![0];
        a *= b;
        assert_eq!(a, c);

        let mut a = sparse_poly![-7];
        let b = sparse_poly![4];
        let c = sparse_poly![-28];
        a *= b;
        assert_eq!(a, c);

        let mut a = sparse_poly![0, 1];
        let b = sparse_poly![4];
        let c = sparse_poly![0, 4];
        a *= b;
        assert_eq!(a, c);

        let mut a = sparse_poly![0, -1];
        let b = sparse_poly![0, 1];
        let c = sparse_poly![0, 0, -1];
        a *= b;
        assert_eq!(a, c);
    }

    #[test]
    fn mul_by_value() {
        let a = sparse_poly![1, 2, 3];
        let b = sparse_poly![2, 4, 6];
        assert_eq!(a * 2, b);

        let mut a = sparse_poly![1, 2, 3];
        let b = sparse_poly![2, 4, 6];
        a *= 2;
        assert_eq!(a, b);
    }

    #[test]
    fn div_by_value() {
        let a = sparse_poly![2, 4, 6];
        let b = sparse_poly![1, 2, 3];
        assert_eq!(a / 2, b);

        let mut a = sparse_poly![2, 4, 6];
        let b = sparse_poly![1, 2, 3];
        a /= 2;
        assert_eq!(a, b);
    }

    #[test]
    fn equality() {
        let a = sparse_poly![1, 0];
        let b = sparse_poly![-1, 0];
        assert!(a != b);
    }
}
