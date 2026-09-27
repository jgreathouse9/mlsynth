% The standard normal CDF in core Octave, standing in for MATLAB's
% Statistics Toolbox normcdf. Exact, not an approximation:
% 0.5*erfc(-1.96/sqrt(2)) = 0.975002104852.
function p = normcdf_(z)
  % change 1: the standard normal CDF in core Octave.
  p = 0.5 * erfc(-z ./ sqrt(2));
end
