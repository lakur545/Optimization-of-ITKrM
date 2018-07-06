function [X,nonzero]=recover_x(y,D,noise,zero_threshold)
% l1-norm heuristic for finding a sparse solution
dic_len=size(D,2);
eps=noise;
delta=zero_threshold;

cvx_begin quiet
  variable x_l1(dic_len)
  minimize( norm( x_l1, 1 ) )
  subject to
    -norm( y-D*x_l1, 2 )+eps>=0;
cvx_end

% number of nonzero elements in the solution (its cardinality or diversity)
nnz = length(find( abs(x_l1) > delta ));
X=x_l1;
nonzero=nnz;
end