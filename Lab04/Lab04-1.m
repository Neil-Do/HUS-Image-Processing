clear
pkg load image

function P = AddPadding(J, n)
  # add padding
  [rows, cols] = size(J);
  P = zeros(rows + 2 * n, cols + 2 * n);
  P(1+n:rows+n, 1+n:cols+n) = J;
endfunction

function R = CutPadding(J, n)
  [rows, cols] = size(J);
  rows -= 2 * n;
  cols -= 2 * n;
  R = J(1 + n:rows+n, 1+n:cols+n);
endfunction

function R = dilation(J, S)
  n = floor(length(S)/2);
  [rows, cols] = size(J);
  J = AddPadding(J, n);
  R = zeros(size(J));
  count = 0;
  for r = 1+n:rows+n
    for c = 1+n:cols+n
      if J(r, c) == 1
        R(r-n:r+n, c-n:c+n) = J(r-n:r+n, c-n:c+n)|S;
      endif
    endfor
  endfor
  R = CutPadding(R, n);
endfunction

# cau 1 - 1
%{
%}
I = imread("coneDetection.jpg");
G = rgb2gray(I);
imwrite(G, "Results/GrayImage.png");
K = (G(:,:) < 120);
imwrite(K, "Results/BinaryImage.png");

# cau 1 - 2
S1 = ones(5);
S2 = ones(3);
T1 = dilation(K, S1);
imwrite(T1, "Results/DilationT1.png");
T2 = dilation(K, S2);
imwrite(T2, "Results/DilationT2.png");
S = T1 = T2;
imwrite(S, "Results/DilationS.png");
%{
  test dilation function
  B = csvread("TestMatrix.csv");
  disp("Here");
  S = ones(3);
  R = dilation(B, S);
  subplot(1, 2, 1);
  imshow(B);
  subplot(1, 2, 2);
  imshow(R);
%}
