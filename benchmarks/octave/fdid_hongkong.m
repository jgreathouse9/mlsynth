% Kathleen T. Li's FDID_Matlab.m and FDID_newR2.m, from the Marketing Science
% replication package (DOI 10.1287/mksc.2022.0212), on her released Hong Kong
% GDP panel. Run under Octave so the benchmark can read the author's own MATLAB
% numbers live rather than transcribing her readme.
%
% The numerics are hers, line for line. Four changes, and nothing else:
%
%   1. normcdf -> 0.5*erfc(-x/sqrt(2)). normcdf lives in MATLAB's Statistics
%      Toolbox and in Octave's statistics package, which is not assumed here.
%      The substitution is the exact same function, not an approximation:
%      0.5*erfc(-1.96/sqrt(2)) = 0.975002104852.
%   2. The two plotting blocks (her lines 82-129) are dropped. They write PDFs
%      and need a display; the benchmark reads numbers.
%   3. The data path arrives as an argument instead of being the working
%      directory's GDP.csv.
%   4. Her FDID_newR2 is a separate file and stays one here, as fdid_newr2.m.
%
% One hazard of hers is left in place and flagged rather than fixed: at her
% FDID_newR2.m line 60, num_c = find(R2final == max(R2final)) returns every
% index attaining the maximum, where the R version's which.max returns the
% first. On a tie the two of her own implementations would diverge. On this
% panel the argmax is unique at 9, so it does not bite, and changing it would
% make this no longer her code.

% fdid_newr2.m and normcdf_.m sit next to this file, mirroring Li's own layout
% (she ships FDID_newR2.m separately), so put this directory on the path.
addpath(fileparts(mfilename('fullpath')));

args = argv();
datafile = 'GDP.csv';
for k = 1:numel(args)
  if strcmp(args{k}, '--data') && k < numel(args)
    datafile = args{k + 1};
  end
end

dataxy = csvread(datafile, 1, 0);   % her line 8: skip the country-name header
datay  = dataxy(:, 1);              % Hong Kong, the treated unit
datax  = dataxy(:, 2:end);          % the 24 controls

t  = size(datay, 1);                % 61
t1 = 44;                            % her line 14, the pre-treatment length
t2 = t - t1;                        % 17
y1 = datay(1:t1, 1);
y2 = datay(t1+1:t, 1);
x  = datax;
no_control = size(x, 2);

control_ID = [1:no_control];
[y1_hat_FDID, y2_hat_FDID, R2final, num_c, select_c] = ...
    fdid_newr2(no_control, control_ID, x, y1, y2, t1, t);

ATT_FDID       = mean(y2 - y2_hat_FDID);
ATT_FDID_per   = 100 * ATT_FDID / mean(y2_hat_FDID);
R2_forward_DID = 1 - (mean((y1 - y1_hat_FDID).^2)) / mean((y1 - mean(y1)).^2);

u1_FDID            = y1 - y1_hat_FDID;
Omega_1_hat_FDID   = (t2/t1) * mean(u1_FDID.^2);
Omega_2_hat_FDID   = mean(u1_FDID.^2);
std_Omega_hat_FDID = sqrt(Omega_1_hat_FDID + Omega_2_hat_FDID);

ATT_std_FDID        = sqrt(t2) * ATT_FDID / std_Omega_hat_FDID;
p_value_forward_DID = 2 * (1 - normcdf_(abs(ATT_std_FDID)));
CI_95_FDID_left     = ATT_FDID - 1.96 * std_Omega_hat_FDID / sqrt(t2);
CI_95_FDID_right    = ATT_FDID + 1.96 * std_Omega_hat_FDID / sqrt(t2);

% her lines 54-79: the conventional DID over all controls
x_DID   = mean(x, 2);
x1_DID  = x_DID(1:t1, :);
x2_DID  = x_DID(t1+1:t, :);
b_DID   = mean(y1 - x1_DID);
y1_DID  = b_DID + x1_DID;
y2_DID  = b_DID + x2_DID;

ATT_DID            = mean(y2 - y2_DID);
ATT_DID_percentage = 100 * ATT_DID / mean(y2_DID);
R2_DID             = 1 - (mean((y1 - y1_DID).^2)) / (mean((y1 - mean(y1)).^2));
u1_DID             = y1 - y1_DID;
Omega_1_hat_DID    = (t2/t1) * mean(u1_DID.^2);
Omega_2_hat_DID    = mean(u1_DID.^2);
std_Omega_hat_DID  = sqrt(Omega_1_hat_DID + Omega_2_hat_DID);
ATT_std_DID        = sqrt(t2) * ATT_DID / std_Omega_hat_DID;
p_value_DID        = 2 * (1 - normcdf_(abs(ATT_std_DID)));
CI_95_DID_left     = ATT_DID - 1.96 * std_Omega_hat_DID / sqrt(t2);
CI_95_DID_right    = ATT_DID + 1.96 * std_Omega_hat_DID / sqrt(t2);

printf('== MATLAB REFERENCE VALUES ==\n');
printf('m_fdid_att\t%.12f\n', ATT_FDID);
printf('m_fdid_att_pct\t%.12f\n', ATT_FDID_per);
printf('m_fdid_r2_pre\t%.12f\n', R2_forward_DID);
printf('m_fdid_n_controls\t%d\n', num_c);
printf('m_fdid_satt\t%.12f\n', ATT_std_FDID);
printf('m_fdid_se\t%.12f\n', std_Omega_hat_FDID / sqrt(t2));
printf('m_fdid_p_value\t%.12f\n', p_value_forward_DID);
printf('m_fdid_ci_lo\t%.12f\n', CI_95_FDID_left);
printf('m_fdid_ci_hi\t%.12f\n', CI_95_FDID_right);
printf('m_did_att\t%.12f\n', ATT_DID);
printf('m_did_att_pct\t%.12f\n', ATT_DID_percentage);
printf('m_did_r2_pre\t%.12f\n', R2_DID);
printf('m_did_satt\t%.12f\n', ATT_std_DID);
printf('m_did_se\t%.12f\n', std_Omega_hat_DID / sqrt(t2));
printf('m_did_p_value\t%.12f\n', p_value_DID);
printf('m_did_ci_lo\t%.12f\n', CI_95_DID_left);
printf('m_did_ci_hi\t%.12f\n', CI_95_DID_right);
printf('== MATLAB SESSION INFO ==\n');
printf('%s\n', version());
