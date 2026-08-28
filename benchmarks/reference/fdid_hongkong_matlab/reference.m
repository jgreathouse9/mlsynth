% Reference run for the `fdid_hongkong` benchmark's inference cells.
%
% Runs Kathleen T. Li's own released MATLAB driver, FDID_Matlab.m (Marketing
% Science replication package for "Forward Difference-in-Differences," DOI
% 10.1287/mksc.2022.0212), unmodified, under GNU Octave. Her Fun_FDID.R --
% which benchmarks/reference/fdid_hongkong/reference.R runs -- returns the
% fit, R^2, selection and ATT but computes no inference at all; the MATLAB
% driver computes the standard error, confidence interval, p-values and the
% standardised ATT. This bundle captures those.
%
% The data is the GDP.csv already vendored under fdid_hongkong/ (verified
% byte-identical to the package's copy), so both bundles pin the same panel.
%
% Run from the repository root:
%   octave --no-gui --quiet benchmarks/reference/fdid_hongkong_matlab/reference.m
here = fileparts(mfilename('fullpath'));
root = fileparts(fileparts(fileparts(here)));
addpath(here);                                   % FDID_newR2.m, normcdf.m shim
set(0, 'defaultfigurevisible', 'off');           % the driver plots two figures
cd(fullfile(root, 'benchmarks', 'reference', 'fdid_hongkong'));  % holds GDP.csv

try
  FDID_Matlab;                                   % the author's driver, verbatim
catch err
  % The driver ends by plotting two figures, and a headless Octave has no
  % graphics toolkit. Every number captured below is computed before that
  % point, so a graphics failure is expected here and harmless. Anything
  % else is a real failure and must not be swallowed.
  if isempty(strfind(err.message, 'graphics toolkit'))
    rethrow(err);
  end
end

% Fail if the driver stopped before defining what this bundle pins,
% so a truncated run can never be captured as a valid reference.
required = {'ATT_FDID', 'ATT_FDID_per', 'R2_forward_DID', 'std_Omega_hat_FDID', ...
            'ATT_std_FDID', 'p_value_forward_DID', 'CI_95_FDID_left', ...
            'CI_95_FDID_right', 'ATT_DID', 'ATT_DID_percentage', 'R2_DID', ...
            'std_Omega_hat_DID', 'ATT_std_DID', 'p_value_DID', ...
            'CI_95_DID_left', 'CI_95_DID_right', 't1', 't2'};
for i = 1:numel(required)
  if ~exist(required{i}, 'var')
    error('reference.m: the driver did not define %s', required{i});
  end
end

se_fdid = std_Omega_hat_FDID / sqrt(t2);
se_did  = std_Omega_hat_DID  / sqrt(t2);

printf('== REFERENCE VALUES ==\n');
printf('fdid_att\t%.10f\n', ATT_FDID);
printf('fdid_att_pct\t%.10f\n', ATT_FDID_per);
printf('fdid_r2_pre\t%.10f\n', R2_forward_DID);
printf('fdid_se\t%.10f\n', se_fdid);
printf('fdid_att_std\t%.10f\n', ATT_std_FDID);
printf('fdid_p_value\t%.10f\n', p_value_forward_DID);
printf('fdid_ci_low\t%.10f\n', CI_95_FDID_left);
printf('fdid_ci_high\t%.10f\n', CI_95_FDID_right);
printf('did_att\t%.10f\n', ATT_DID);
printf('did_att_pct\t%.10f\n', ATT_DID_percentage);
printf('did_r2_pre\t%.10f\n', R2_DID);
printf('did_se\t%.10f\n', se_did);
printf('did_att_std\t%.10f\n', ATT_std_DID);
printf('did_p_value\t%.10f\n', p_value_DID);
printf('did_ci_low\t%.10f\n', CI_95_DID_left);
printf('did_ci_high\t%.10f\n', CI_95_DID_right);
printf('t1\t%d\n', t1);
printf('t2\t%d\n', t2);
printf('== SESSION INFO ==\n');
printf('GNU Octave version %s\n', version());
