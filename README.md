# Polynomials
A minimal implementation of polynomial arithmetic, that does not rely on the standard library. (i.e. can be built with #![no_std])

# Documentation
Detailed documentation can be found [here](https://docs.rs/polynomials).

# Usage
A `Polynomial` is a vector of coefficients. To create a new polynomial:
```
let mut p = Polynomial::new();
```

You can add coefficients to this polynomial, just like adding elements to a vector:
```
// 3x^2 + 2x + 1
p.push(1);
p.push(2);
p.push(3);
```

Or you can instantiate a vector with the `poly!` macro, similar to a vector:
```
let p = poly![1, 2, 3];
```

You can multiply or divide a polynomial by a constant:
```
let new = p * 2;
assert_eq!(new, poly![2, 4, 6]);
assert_eq!(new / 2, poly![1, 2, 3]);
```

You can multiply a polynomial by another polynomial:
```
// (x + 1)(x - 1) = x^2 - 1
let a = poly![1, 1]; // x + 1
let b = poly![1, -1]; // x - 1
assert_eq!(a * b, poly![-1, 0, 1]);
```

And of course, you can evaluate a polynomial at some value:
```
let p = poly![1, 1]; // x + 1
assert_eq!(p.eval(7).unwrap(), 8); // 1*7 + 1
```

