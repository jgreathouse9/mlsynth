% Kathleen T. Li's FDID_newR2.m, from the Marketing Science replication
% package (DOI 10.1287/mksc.2022.0212), verbatim on the numerics. See
% fdid_hongkong.m's header for the one hazard of hers left in place.
function [y1h, y2h, R2final, num_c, select_c] = ...
         fdid_newr2(no_control, control_ID, x, y1, y2, t1, t)
  % Her FDID_newR2.m, verbatim.
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
  if length(first_c) > 1
    first_c = first_c(1, 1);
  end
  select_c(1, 1) = first_c;

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

  num_c = find(R2final == max(R2final));
  num_c = num_c(1);          % her line 60 returns every argmax; see the header
  control = x(1:end, select_c(1:num_c));
  x1_forward_DID = mean(control(1:t1, :), 2);
  x2_forward_DID = mean(control((t1+1):t, :), 2);
  beta_forward_DID = mean(y1 - x1_forward_DID);
  y1h = beta_forward_DID + x1_forward_DID;
  y2h = beta_forward_DID + x2_forward_DID;
end
