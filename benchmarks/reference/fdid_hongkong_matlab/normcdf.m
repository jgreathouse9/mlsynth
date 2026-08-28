function p = normcdf(x)
%NORMCDF Standard normal cumulative distribution function.
%   Shim for GNU Octave, whose base install does not ship NORMCDF (it lives
%   in the statistics package, and in MATLAB in the Statistics Toolbox).
%   FDID_Matlab.m calls normcdf(x) with the standard normal only, and this
%   is that CDF in closed form -- Phi(x) = erfc(-x/sqrt(2))/2 -- so it is
%   exact to machine precision, not an approximation. Vendored so the
%   author's driver runs unmodified.
  p = 0.5 * erfc(-x ./ sqrt(2));
end
