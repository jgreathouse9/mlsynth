% Li & Shankar (2023) "A Two-Step Synthetic Control Approach for Estimating
% Causal Effects of Marketing Events", Management Science -- the R-side
% reference for Tables 2 to 5, transcribed from the replication package's
% TSSC_Tables2_5_Test_MSE.m and run under GNU Octave.
%
% Three departures from the shipped MATLAB, all of them forced:
%
%   lsqlin      Octave ships it only in the optim package, which the suite
%               would then have to install. The shim below is the same
%               problem -- minimise ||C x - d||^2 subject to equalities and
%               bounds -- handed to core Octave's qp.
%   rng shuffle The original reseeds from the clock on every replication and
%               every subsampling draw, so its own ReadMe says the numbers
%               "will be similar to but not be exactly what is in the paper".
%               A cross-validation needs a run that repeats, so the stream is
%               seeded from --seed.
%   datasample  Statistics-toolbox only; replaced by randi, which is what
%               datasample does with replacement.
%
% Nothing else moves: the three-factor DGP, the four constrained fits, the two
% restriction statistics, the joint statistic and its subsampling variance, and
% the two-sided coverage indicators are the authors'.
%
% Writes, into --out:
%   summary_dgp<k>.csv     per replication: the five ATTs, the three test
%                          statistics, and coverage at the five levels
%   panels_dgp<k>.csv      the first --dump panels, long, for the exact seam
%
% Usage:
%   octave-cli benchmarks/octave/tssc_tables2_5.m --out DIR [--nr N] [--nb M]
%              [--t1 T] [--dump K] [--seed S]

function tssc_tables2_5()
  args = argv();
  outdir = argval(args, '--out', 'tssc_ref');
  nr     = str2double(argval(args, '--nr', '200'));
  nb     = str2double(argval(args, '--nb', '400'));
  t1     = str2double(argval(args, '--t1', '80'));
  t2     = str2double(argval(args, '--t2', '20'));
  mm     = str2double(argval(args, '--m', '20'));
  ndump  = str2double(argval(args, '--dump', '20'));
  seed   = str2double(argval(args, '--seed', '20260921'));
  mkdir(outdir);

  n = 11;                      % 1 treated + 10 controls
  t = t1 + t2;
  c1 = 1; c2 = 1;
  C_TE = 0;                    % Tables 2-5 run under no treatment effect

  % H_0 : both restrictions.  H_0a : weights sum to one.  H_0b : zero intercept.
  Rt  = [1, zeros(1, n-1); 0, ones(1, n-1)];  qt  = [0; 1];
  R1t = [0, ones(1, n-1)];                    q1t = 1;
  R2t = [1, zeros(1, n-1)];                   q2t = 0;

  levels = [0.50 0.80 0.90 0.95 0.99];

  for dgp = 1:3
    a0 = 0; b0 = 0;
    if dgp == 2, b0 = 0.1; end      % H_0a violated
    if dgp == 3, a0 = 0.5; end      % H_0b violated

    rand('state', seed + 1000*dgp);
    randn('state', seed + 1000*dgp);

    b1 = (c1 + b0) * ones(3, 1);
    b2 = c2 * ones(3, n-1);
    b  = [b1, b2];

    rows = zeros(nr, 5 + 3 + 15);
    dump = [];
    for ii = 1:nr
      [y, Del_TE] = draw_panel(t, n, b, a0, C_TE);
      x1 = y(1:t1, 2:n);  y1 = y(1:t1, 1);
      x2 = y(t1+1:t, 2:n); y2 = y(t1+1:t, 1) + Del_TE(t1+1:t);
      XX1 = [ones(t1,1), x1];  XX2 = [ones(t2,1), x2];

      % MSC(c): free intercept, non-negative slopes
      lb = zeros(n,1); lb(1) = -Inf;
      b_MSC  = lsqlin_qp(XX1, y1, [], [], lb, []);
      att_MSC = mean(y2 - XX2*b_MSC);
      % MSC(b): no intercept, non-negative slopes
      b_MSCb = lsqlin_qp(XX1(:,2:end), y1, [], [], zeros(n-1,1), []);
      att_MSCb = mean(y2 - XX2(:,2:end)*b_MSCb);
      % SC: no intercept, slopes on the simplex
      b_SC   = lsqlin_qp(XX1(:,2:end), y1, ones(1,n-1), 1, zeros(n-1,1), ones(n-1,1));
      att_SC = mean(y2 - XX2(:,2:end)*b_SC);
      % MSC(a): free intercept, slopes on the simplex
      lba = zeros(n,1); lba(1) = -Inf; upa = ones(n,1); upa(1) = Inf;
      Aeqa = ones(1,n); Aeqa(1) = 0;
      b_MSCa = lsqlin_qp(XX1, y1, Aeqa, 1, lba, upa);
      att_MSCa = mean(y2 - XX2*b_MSCa);

      d1t = R1t*b_MSC - q1t;  d2t = R2t*b_MSC - q2t;  dt = Rt*b_MSC - qt;
      test1 = t1*(d1t'*d1t);  test2 = t1*(d2t'*d2t);

      % subsampling: m rows with replacement, recentred statistics
      ZZ1 = [XX1, y1];
      bm = zeros(n, nb); t1s = zeros(nb,1); t2s = zeros(nb,1); bsd = zeros(nb, n);
      for g = 1:nb
        idx = randi(t1, mm, 1);
        ZZm = ZZ1(idx, :);
        bmg = lsqlin_qp(ZZm(:,1:n), ZZm(:,n+1), [], [], lb, []);
        bm(:,g) = bmg;
        t1s(g) = mm*((R1t*(bmg - b_MSC))^2);
        t2s(g) = mm*((R2t*(bmg - b_MSC))^2);
        bsd(g,:) = sqrt(mm)*(bmg - b_MSC)';
      end
      V_hat = inv(Rt*(mm*(bsd'*bsd)/nb)*Rt');
      Js = zeros(nb,1);
      for g = 1:nb
        ds = Rt*(bm(:,g) - b_MSC);
        Js(g) = mm*ds'*V_hat*ds;
      end
      J_test = t1*dt'*V_hat*dt;

      cov_J  = coverage(J_test, sort(Js),  nb, levels);
      cov_1a = coverage(test1,  sort(t1s), nb, levels);
      cov_1b = coverage(test2,  sort(t2s), nb, levels);

      rows(ii,:) = [att_SC, att_MSCa, att_MSCb, att_MSC, NaN, ...
                    J_test, test1, test2, cov_J, cov_1a, cov_1b];
      if ii <= ndump
        unit = kron((1:n)', ones(t,1));
        dump = [dump; [repmat(ii, t*n, 1), unit, repmat((1:t)', n, 1), y(:)]];
      end
    end
    hdr = ['att_SC,att_MSCa,att_MSCb,att_MSCc,att_TSSC,J_test,test_H0a,test_H0b,' ...
           'covJ50,covJ80,covJ90,covJ95,covJ99,' ...
           'covA50,covA80,covA90,covA95,covA99,' ...
           'covB50,covB80,covB90,covB95,covB99'];
    write_csv(fullfile(outdir, sprintf('summary_dgp%d.csv', dgp)), hdr, rows);
    write_csv(fullfile(outdir, sprintf('panels_dgp%d.csv', dgp)), 'rep,unit,time,y', dump);
    printf('dgp %d  size/power H0 %.3f  H0a %.3f  H0b %.3f  (nr=%d)\n', ...
           dgp, 1-mean(rows(:,12)), 1-mean(rows(:,17)), 1-mean(rows(:,22)), nr);
    fflush(stdout);
  end
  printf('wrote %s\n', outdir);
end

function [y, Del_TE] = draw_panel(t, n, b, a0, C_TE)
  ss = sqrt(3);
  u1 = 2*ss*rand(t,1) - ss;  u2 = 2*ss*rand(t,1) - ss;
  u3 = 2*ss*rand(t,1) - ss;  u4 = 0.5*(2*ss*rand(t,1) - ss);
  f1 = zeros(t,1);
  for k = 1:(t-1)
    f1(k+1) = 0.2*(k+1) - 0.8*sqrt(k+1) + 0.8*f1(k) + u1(k);   % nonlinear trend
  end
  f2 = zeros(t,1);
  for k = 1:(t-2)
    f2(k+1) = -0.6*f2(k) + u2(k+1) + 0.8*u2(k);                % ARMA(1,1)
  end
  f3 = zeros(t,1);
  for k = 1:(t-2)
    f3(k+2) = u3(k+2) + 0.9*u3(k+1) + 0.4*u3(k);               % MA(2)
  end
  z = zeros(t,1);
  for k = 1:(t-1)
    z(k+1) = 0.5*z(k) + u4(k+1);
  end
  Del_TE = C_TE*(exp(z)./(1+exp(z)) + 1);
  f = [f1, f2, f3];
  y = ones(t,n);
  eps = randn(t,n);
  for k = 1:n
    y(:,k) = ones(t,1) + f*b(:,k) + eps(:,k);
  end
  y(:,1) = y(:,1) + a0;   % the treated unit's intercept shift under DGP3
end

function c = coverage(stat, sorted_s, nb, levels)
  c = zeros(1, numel(levels));
  for j = 1:numel(levels)
    a = (1 - levels(j))/2;
    lo = sorted_s(max(1, round(a*nb)));
    hi = sorted_s(max(1, round((1-a)*nb)));
    c(j) = (stat > lo) && (stat < hi);
  end
end

function x = lsqlin_qp(C, d, Aeq, beq, lb, ub)
  % min ||C x - d||^2  s.t.  Aeq x = beq,  lb <= x <= ub, via core qp.
  p = columns(C);
  H = C'*C;  q = -C'*d;
  H = (H + H')/2;
  if isempty(lb), lb = -Inf(p,1); end
  if isempty(ub), ub =  Inf(p,1); end
  x0 = max(min(pinv(C)*d, min(ub, 1e6)), max(lb, -1e6));
  if isempty(Aeq)
    x = qp(x0, H, q, [], [], lb, ub);
  else
    x = qp(x0, H, q, Aeq, beq, lb, ub);
  end
  if isempty(x), x = x0; end
end

function v = argval(args, flag, default)
  v = default;
  for i = 1:numel(args)-1
    if strcmp(args{i}, flag), v = args{i+1}; return; end
  end
end

function write_csv(path, header, data)
  fid = fopen(path, 'w');
  fprintf(fid, '%s\n', header);
  fclose(fid);
  if ~isempty(data)
    dlmwrite(path, data, '-append', 'precision', '%.10g');
  end
end

tssc_tables2_5();
