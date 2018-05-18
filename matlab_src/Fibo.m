function [ r ] = Fibo( n )
%return the n-th of the Fibonacci sequence
tic
a = 1;
b = 1;
for i = 2:n
    tmp = b;
    b = a+b;
    a = tmp;
end
r = b;
toc
end

% function[r] = Fibo(n)
% tic
% M = [0 1;1 1];
% init = [1 1];
% res = M^(n-1)*init';
% r = res(2)
% toc
% end
