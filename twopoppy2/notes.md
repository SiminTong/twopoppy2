Notes to myself!

When setting the characteristic radius Rc, don't forget to check
whether the radial grid is set properly. 

If the grid is too large and the characteristic radius is too small,
(e.g.: rmin = 0.1*au, rmax=5000 * au, nr=2000, rc = 5 au),
the gas surface density in the outermost cells will be extremely small,
and becomes zero due to the precision. This will cause problems in 
matrix computation and fail to compute the flux (modules written 
in FORTRAN), which will eventually lead to false timesteps.
