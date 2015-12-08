GradOpt: Scala Convex Optimizations
===================================

Provdes implementations for widely-used convex optimization algorithms.
Key differences from existing packages are preference of functional
programming paradigms where possible, such as
* `Stream` (lazy list) implementations of iterative procedures such as
  bracketing, line search, and gradient descent
* Failures expressed at the type-level using `Option`s
* Modular compositional architecture enabling clean separation of
  bracketing, line search, and iterative decision variable update
  routines and glued together through

Gradient Algorithms:
* Steepest-Descent
* Conjugate Gradient with Fletcher-Reeves formula

Line Search Algorithms:
* Cubic interpolation with zoom satisfying Strong Wolfe Conditions
* Exact step-size for quadratic forms

WishList
-------

[ ] More conjugate gradient formulas
[ ] More line search interpolation methods
[ ] Conjugate-gradient preconditioning
[ ] Nelder-Mead gradient-free optimization (though this may require name
change)
[ ] Plotting utilities for 1D/2D optimization problems

Installation
-----------

In `build.sbt`
```sbt
TODO: add SBT installation commands
```

Usage
-----

```scala
TODO: provide simple usage example
```
