use core::fmt::Debug;
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
use sp_std::collections::btree_map::{BTreeMap};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparsePolynomial<T>(Vec<(u64,T)>);

impl<T> SparsePolynomial<T> {
    pub fn new() -> SparsePolynomial<T> {
        SparsePolynomial(Vec::<(u64,T)>::new())
    }

    pub fn into_map(&self) -> BTreeMap<u64,T> where T: Add<Output = T> + Clone {
        let mut map: BTreeMap<u64,T> = BTreeMap::new();
        for i in 0..self.0.len() {
            match map.get_mut(&self[i].0) {
                Some(value) => { *value = value.clone() + self[i].1.clone(); },
                None => { map.insert(self[i].0, self[i].1.clone()); },
            };
        }

        map
    }

    pub fn push(&mut self, value: (u64,T)) {
        self.0.push(value);
    }

    pub fn pop(&mut self) -> Option<(u64,T)> {
        self.0.pop()
    }

    pub fn degree(&self) -> usize
    where
        T: Sub<T, Output = T> + Eq + Copy,
    {
        let mut deg = 0;
        for i in 0..self.0.len() {
            if self.0[i].0 > deg {
                deg = self.0[i].0;
            }
        }
        deg as usize
    }

    pub fn eval(&self, x: T) -> Option<T>
    where
        T: AddAssign + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Copy,
        T: MulAssign + Div<Output = T> + Debug,
    {
        if self.0.len() == 0 {
            None
        } else {
            let zero: T = x - x;
            let mut total: T = zero;

            for (exponent, coefficient) in self.into_map().into_iter() {
                total += coefficient * self.pow(x, exponent);
            }

            Some(total)
        }
    }

    pub fn pow(&self, x: T, exponent: u64) -> T where T: MulAssign + Div<Output = T> + Copy {
        if exponent == 0 {
            x / x
        } else {
            let mut temp = x;
            for _ in 1..exponent {
                temp *= x;
            }

            temp
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = &(u64,T)> {
        self.0.iter()
    }

    pub fn into_iter(self) -> impl IntoIterator<Item = (u64,T), IntoIter = IntoIter<(u64,T)>> {
        self.0.into_iter()
    }
}

impl<T> From<Vec<(u64,T)>> for SparsePolynomial<T> {
    fn from(v: Vec<(u64,T)>) -> Self {
        SparsePolynomial(v)
    }
}

impl<T> Into<Vec<(u64,T)>> for SparsePolynomial<T> {
    fn into(self) -> Vec<(u64,T)> {
        self.0
    }
}

impl<T, I: SliceIndex<[(u64,T)]> + SliceIndex<[(u64, T)]>> Index<I> for SparsePolynomial<T> {
    type Output = I::Output;

    fn index(&self, index: I) -> &Self::Output {
        &self.0[index]
    }
}

impl<T, I: SliceIndex<[(u64,T)]> + SliceIndex<[(u64, T)]> + SliceIndex<[(u64, T)]>> IndexMut<I> for SparsePolynomial<T> {
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl<T: Add<Output = T>> Add for SparsePolynomial<T>
where
    T: Add + Copy + Clone,
    Self: AddAssign,
{
    type Output = Self;

    fn add(mut self, other: Self) -> Self::Output {
        self += other;
        self
    }
}

impl<T> AddAssign for SparsePolynomial<T>
where
    T: Add<Output = T> + Copy + PartialEq + Sub<Output = T>,
{
    fn add_assign(&mut self, rhs: Self) {
        let mut poly_map = self.into_map();
        let rhs_map = rhs.into_map();

        for (k, v) in rhs_map.into_iter() {
            match poly_map.get_mut(&k) {
                Some(value) => { *value = *value + v; },
                None => { poly_map.insert(k, v); },
            };
        }

        self.0 = poly_map
            .into_iter()
            .filter(|elt| elt.1 != elt.1 - elt.1)
            .collect::<Vec<_>>();
    }
}

impl<T: Sub<Output = T>> Sub for SparsePolynomial<T>
where
    T: Sub + Neg<Output = T> + Copy + Clone,
    Self: SubAssign,
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
    T: Add<Output = T> + PartialEq,
{
    fn sub_assign(&mut self, rhs: Self) {
        let mut poly_map = self.into_map();
        let rhs_map = rhs.into_map();

        for (k, v) in rhs_map.into_iter() {
            match poly_map.get_mut(&k) {
                Some(value) => { *value = *value - v; }
                None => { poly_map.insert(k, -v); },
            };
        }

        self.0 = poly_map
            .into_iter()
            .filter(|elt| elt.1 != elt.1 - elt.1)
            .collect::<Vec<_>>();
    }
}

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
            self[i].1 *= rhs;
        }
    }
}

impl<T> Mul for SparsePolynomial<T>
where
    T: Mul<Output = T> + AddAssign + Sub<Output = T>,
    T: Copy + Clone,
    Self: MulAssign
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
    T: Add<Output = T> + Copy + Clone + PartialEq,
{
    fn mul_assign(&mut self, rhs: Self) {
        let poly_map = self.into_map();
        let rhs_map = rhs.into_map();
        let mut new_map: BTreeMap<u64, T> = BTreeMap::new();

        // One of the vectors must be non-empty
        if self.0.len() > 0 || rhs.0.len() > 0 {
            // Since core::num does not provide the `Zero()` trait
            // this hack lets us calculate zero from any generic
            let zero = if self.0.len() > 0 {
                self[0].1 - self[0].1
            } else {
                rhs[0].1 - rhs[0].1
            };

            // Calculate product
            for (k1,v1) in poly_map.into_iter() {
                for (k2,v2) in rhs_map.clone().into_iter() {
                    match new_map.get_mut(&(k1 + k2)) {
                        Some(value) => { *value = *value + (v1 * v2); },
                        None => { new_map.insert(k1 + k2, v1 * v2); },
                    }
                }
            }

            self.0 = new_map
                .into_iter()
                .filter(|elt| elt.1 != elt.1 - elt.1)
                .collect::<Vec<_>>();
        }
    }
}

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
            self[i].1 /= rhs;
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

        for (l, r) in self.0.iter().zip(other.0.iter()) {
            let (lk, lv) = l;
            let (rk, rv) = r;
            if lk != rk || lv != rv {
                return false
            }
        }
        true
    }
}
impl<T> Eq for SparsePolynomial<T> where T: Sub<T, Output = T> + Eq + Copy {}

#[macro_export]
macro_rules! sparse_poly {
    ($($args:tt)*) => (
         $crate::Polynomial::from(vec![$($args)*])
     );
}
