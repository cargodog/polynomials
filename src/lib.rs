use core::cmp::PartialEq;
use core::convert::From;
use core::ops::Neg;
use core::ops::{Add, AddAssign};
use core::ops::{Div, DivAssign};
use core::ops::{Index, IndexMut};
use core::ops::{Mul, MulAssign};
use core::ops::{Sub, SubAssign};
use core::slice::SliceIndex;

extern crate alloc;
use alloc::vec::IntoIter;

#[derive(Debug, Clone)]
pub struct Polynomial<T>(Vec<T>);

impl<T> Polynomial<T> {
    pub fn new() -> Polynomial<T> {
        Polynomial(Vec::<T>::new())
    }

    pub fn push(&mut self, value: T) {
        self.0.push(value);
    }

    pub fn pop(&mut self) -> Option<T> {
        self.0.pop()
    }

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
        if self.0.len() < rhs.0.len() {
            for i in self.0.len()..rhs.0.len() {
                self.push(rhs[i]);
            }
        }
        for i in 0..self.0.len() {
            self[i] = self[i] + rhs[i];
        }
    }
}

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
        if self.0.len() < rhs.0.len() {
            for i in self.0.len()..rhs.0.len() {
                self.push(-rhs[i]);
            }
        }
        for i in 0..self.0.len() {
            self[i] = self[i] - rhs[i];
        }
    }
}

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

impl<T> Mul for Polynomial<T>
where
    T: Mul<Output = T> + AddAssign + Sub<Output = T>,
    T: Copy + Clone,
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut new = Self::new();

        // One of the vectors must be non-empty
        if self.0.len() > 0 || rhs.0.len() > 0 {
            // Since core::num does not provide the `Zero()` trait
            // this hack lets us calculate zero from any generic
            let zero = if self.0.len() > 0 {
                self[0] - self[0]
            } else {
                rhs[0] - rhs[0]
            };

            // New vector with size M + N - 1
            new.0.resize(self.0.len() + rhs.0.len() - 1, zero);

            // Calculate product
            for i in 0..self.0.len() {
                for j in 0..rhs.0.len() {
                    new[i + j] += self[i] * rhs[j];
                }
            }
        }
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
        for i in 0..degree {
            if self[i] != other[i] {
                return false;
            }
        }
        true
    }
}
impl<T> Eq for Polynomial<T> where T: Sub<T, Output = T> + Eq + Copy {}

