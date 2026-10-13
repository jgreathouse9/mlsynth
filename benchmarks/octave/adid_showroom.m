% Li and Van den Bulte's ADID script, run under Octave.
%
% Source: Showroom_generated_data_ADID_DID.m from the authors' replication
% material, vendored verbatim beside this file in
% benchmarks/reference/adid_showroom/.
%
% Four changes and no others, so the comparison is against their arithmetic:
%   1. the data path and the pre-period length are arguments, where the script
%      hardcodes a filename and picks a city with an if/elseif;
%   2. the two plotting blocks and their saveas calls are dropped;
%   3. the results are printed as tab-separated key/value pairs;
%   4. csvread is replaced by dlmread with a one-row offset, since the vendored
%      CSV carries a header row the original files do not.
% Everything between is their lines 26 to 68, unaltered.

1;

function adid_showroom(datafile, t1)
  new = dlmread(datafile, ',', 1, 0);     % change (4)
  t = rows(new);
  t2 = t - t1;
  datay = new(:, 1);
  datax = new(:, 2:end);
  const = ones(t, 1);
  x = [const, datax];
  x1 = x(1:t1, :);
  x2 = x(t1+1:t, :);
  y = datay;
  y1 = datay(1:t1, 1);
  y2 = datay(t1+1:t, 1);

  % ---- their DID block, lines 29 to 40 ----------------------------------
  x10 = x1(:, 2:end);
  x20 = x2(:, 2:end);
  x1_DID = [mean(x10, 2)];
  beta_DID = mean(y1 - x1_DID);
  r_2_DID = 1 - mean((y1 - beta_DID - x1_DID).^2) / mean((y1 - mean(y1)).^2);
  x2_DID = mean(x20, 2);
  y1_DID = beta_DID + x1_DID;
  y2_DID = beta_DID + x2_DID;
  ATT_DID = mean(y2 - y2_DID);
  ATT_DID_per = 100 * ATT_DID / mean(y2_DID);

  % ---- their ADID block, lines 46 to 68 ---------------------------------
  x10 = datax(1:t1, :);
  x20 = datax(1+t1:t, :);
  x1_ADID = [ones(size(x10, 1), 1), mean(x10, 2)];
  x2_ADID = [ones(size(x20, 1), 1), mean(x20, 2)];
  b_ADID = (x1_ADID' * x1_ADID) \ (x1_ADID' * y1);
  y1_ADID = x1_ADID * b_ADID;
  y2_ADID = x2_ADID * b_ADID;
  ATT_ADID = mean(y2 - y2_ADID);
  ATT_ADID_per = 100 * ATT_ADID / mean(y2_ADID);

  e1_ADID = y1 - y1_ADID;
  sigma2_ADID = mean(e1_ADID.^2);
  eta_ADID = mean(x2_ADID)';
  psi_ADID = x1_ADID' * x1_ADID / t1;
  Omega_1_ADID = sigma2_ADID * eta_ADID' * inv(psi_ADID) * eta_ADID;
  Omega_2_ADID = sigma2_ADID;
  Omega_ADID = (t2/t1) * Omega_1_ADID + Omega_2_ADID;
  ATT_ADID_std = sqrt(t2) * ATT_ADID / sqrt(Omega_ADID);

  % ---- change (3): print instead of plot --------------------------------
  printf('t\t%d\n', t);
  printf('t1\t%d\n', t1);
  printf('t2\t%d\n', t2);
  printf('delta1\t%.12f\n', b_ADID(1));
  printf('delta2\t%.12f\n', b_ADID(2));
  printf('adid_att\t%.12f\n', ATT_ADID);
  printf('adid_att_pct\t%.12f\n', ATT_ADID_per);
  printf('adid_sigma2\t%.12f\n', sigma2_ADID);
  printf('adid_omega1\t%.12f\n', Omega_1_ADID);
  printf('adid_omega\t%.12f\n', Omega_ADID);
  printf('adid_std_stat\t%.12f\n', ATT_ADID_std);
  printf('did_intercept\t%.12f\n', beta_DID);
  printf('did_att\t%.12f\n', ATT_DID);
  printf('did_att_pct\t%.12f\n', ATT_DID_per);
  printf('did_r2_pre\t%.12f\n', r_2_DID);
  for i = 1:t
    printf('adid_cf\t%d\t%.12f\n', i, [y1_ADID; y2_ADID](i));
  end
  for i = 1:t
    printf('did_cf\t%d\t%.12f\n', i, [y1_DID; y2_DID](i));
  end
end

args = argv();
adid_showroom(args{1}, str2num(args{2}));
