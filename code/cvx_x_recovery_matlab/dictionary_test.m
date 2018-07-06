close all
clear all
clc
%%Import data section


%%
Y=[1,3,4;2,4,1];      % The set of all your ys, where each y is a coloum

D=[1,5,7,9;1,3,9,5];  % The dictionary, where is coloum is an atom

noise=10^-3;          % this we have to play with, it simply our expected 
                      % error size in the conversion between x and y

zero_threshold=10^-8; % How small does a number have to be to be seen as zero

itr=size(Y,2);        % simply how many y's do we have
dic_len=size(D,2);    % how many atoms

nnz_ar=zeros(itr,1);  % pre locate information regardins sparsness
x_ar=zeros(dic_len,itr); % pre locate storage for the xs
%%
for i=1:itr
[x_ar(:,i),nnz_ar(i)]=recover_x(Y(:,i),D,noise,zero_threshold);
end