% Value-for-value cross-check of port.rpca_hqf against the authors' RPCA_HQF.m.
%
% Run from a directory holding the authors' files at ca4d23a
% (https://github.com/bestzywang/Robust-PCA-via-non-convex-half-quadratic-regularization):
%
%   octave-cli --eval "addpath('<that dir>'); run('octave_crosscheck.m')"
%
% For each (SNR, run) it builds the Table-1 problem, seeds rand so the U drawn
% inside RPCA_HQF is known, calls the authors' function unmodified, and saves
% the inputs and output to crosscheck/snr<S>_run<R>.mat. crosscheck.py replays
% each case through port.rpca_hqf from the same M_noise and U0 and reports the
% largest entrywise difference in X.
pkg load statistics;
outdir = 'crosscheck';
if ~exist(outdir, 'dir'), mkdir(outdir); end
m = 500; n = m; rak = m / 50;
for snr = [3 6 9 12 15]
  for run = 1:5
    randn('state', 1000 * snr + run); rand('state', 1000 * snr + run);
    U = randn(m, rak); V = randn(n, rak); M = U * V';
    M_noise = M + Gaussian_noise(M, 'GM', snr);
    state = rand('state');
    U0 = rand(m, rak);            % what RPCA_HQF's first rand(m, rak) will draw
    rand('state', state);
    X = RPCA_HQF(M_noise, rak, 100, 3, 0.001);
    rmse = norm(X - M, 'fro') / sqrt(m * n);
    save('-v7', sprintf('%s/snr%d_run%d.mat', outdir, snr, run), 'M_noise', 'U0', 'X', 'U', 'V', 'rmse');
  end
end
