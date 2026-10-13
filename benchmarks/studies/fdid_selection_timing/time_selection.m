% Times the forward-selection loop in Kathleen T. Li's released FDID_newR2.m,
% run under Octave. Only the selection is timed: her subroutine selects and then
% fits, and the fit is a constant that does not scale in the donor count.
%
% Usage: octave-cli --no-gui time_selection.m --data panel.csv --t1 44 --reps 5
%   panel.csv: column 1 the treated unit, the rest controls, one row per period.
1;

args = argv();
datafile = ''; t1 = 0; reps = 1;
for k = 1:numel(args)
  if strcmp(args{k}, '--data'), datafile = args{k+1}; end
  if strcmp(args{k}, '--t1'),   t1 = str2double(args{k+1}); end
  if strcmp(args{k}, '--reps'), reps = str2double(args{k+1}); end
end

dat = csvread(datafile, 1, 0);
y = dat(:, 1);
x = dat(:, 2:end);
t = size(dat, 1);
y1 = y(1:t1);
no_control = size(x, 2);
control_ID = 1:no_control;

function [select_c, num_c] = select_once(no_control, control_ID, x, y1, t1)
  % her FDID_newR2.m lines 9-57, the selection only
  R2 = zeros(no_control, 1);
  R2final = zeros(no_control, 1);
  select_c = zeros(1, no_control);
  for j = 1:no_control
    x1_DID = x(1:t1, control_ID(j));
    beta_DID = mean(y1 - x1_DID);
    y1_hat_DID = beta_DID + x1_DID;
    R2(j, :) = 1 - (mean((y1 - y1_hat_DID).^2)) / (mean((y1 - mean(y1)).^2));
  end
  R2final(1, :) = max(R2);
  first_c = find(R2 == max(nonzeros(R2)));
  select_c(1, 1) = first_c(1);
  for k = 2:no_control
    left = setdiff(control_ID, select_c);
    R2 = zeros(length(left), 1);
    for jj = 1:length(left)
      control_1 = x(1:t1, [nonzeros(select_c)', left(jj)]);
      x1_f_DID = mean(control_1, 2);
      beta_f_DID = mean(y1 - x1_f_DID);
      y1_hat_f_DID = beta_f_DID + x1_f_DID;
      R2(jj, :) = 1 - (mean((y1 - y1_hat_f_DID).^2)) / (mean((y1 - mean(y1)).^2));
    end
    R2final(k, :) = max(R2);
    sel = left(find(R2 == max(nonzeros(R2))));
    select_c(1, k) = sel(1);
  end
  nc = find(R2final == max(R2final));
  num_c = nc(1);
end

[sel, num_c] = select_once(no_control, control_ID, x, y1, t1);
elapsed = zeros(reps, 1);
for r = 1:reps
  tic;
  select_once(no_control, control_ID, x, y1, t1);
  elapsed(r) = toc;
end

printf('impl\tMATLAB/Octave (Li FDID_newR2.m selection loop)\n');
printf('num_c\t%d\n', num_c);
printf('selected\t%s\n', strjoin(arrayfun(@(v) num2str(v), sel(1:num_c), ...
                                          'UniformOutput', false), ','));
printf('median_seconds\t%.6f\n', median(elapsed));
printf('min_seconds\t%.6f\n', min(elapsed));
printf('reps\t%d\n', reps);
