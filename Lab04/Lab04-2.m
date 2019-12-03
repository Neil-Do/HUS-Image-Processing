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


function R = erosion(T, W)

  n = floor(length(W)/2);
  [rows, cols] = size(T);
  T = AddPadding(T, n);

  # binary matrix for Ws element > 0
  flags = W > 0;

  R = zeros(size(T));
  for r = 1+n:rows+n
    for c = 1+n:cols+n
      if min(T(r-n:r+n, c-n:c+n)(flags) > 0) # StructElement nam trong mien T(p) > 0 voi moi p

        # tim gia tri min cho phep erosion tren mien hang xom
        min_value = min((T(r-n:r+n, c-n:c+n) - W)(flags));

        # loai tru truong hop min < 0
        R(r, c) = max(0, min_value);

      endif
    endfor
  endfor
  R = CutPadding(R, n);
endfunction


I = imread("fence.png");
T = rgb2gray(I);
imwrite(T, "Results/Bai02/GrayImage.png")
W = zeros(101);
W(:, 51) = ones(101, 1);
W(51, :) = ones(1, 101);

K = uint8(erosion(T, W));
B = K > 200;

%{
%}
# su dung thu vien
K1 = imerode(T, W);
B1 = K1 > 200;

# Ham tu dung
subplot(2, 3, 1);
imshow(I);
subplot(2, 3, 2);
imshow(K);
subplot(2, 3, 3);
imshow(B);

# Thu vien
subplot(2, 3, 4);
imshow(I);
subplot(2, 3, 5);
imshow(K1);
subplot(2, 3, 6);
imshow(B1);
