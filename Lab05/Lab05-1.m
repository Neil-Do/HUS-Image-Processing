clear
pkg load image

function P = AddPadding(J, n)
  # add padding
  [rows, cols] = size(J);
  P = double(ones(rows + 2 * n, cols + 2 * n));
  P(1+n:rows+n, 1+n:cols+n) = J;
endfunction

function R = CutPadding(J, n)
  [rows, cols] = size(J);
  rows -= 2 * n;
  cols -= 2 * n;
  R = J(1 + n:rows+n, 1+n:cols+n);
endfunction


function R = median(J, S)
  n = floor(length(S)/2);
  [rows, cols] = size(J);
  J = AddPadding(J, n);
  R = double(ones(size(J)));
  for r = 1+n:rows+n
    for c = 1+n:cols+n
      count = 0;
      unique_v = unique(J(r - n: r + n,c - n: c + n));
      for i = 1:length(unique_v)
        count += sum(S(J(r - n: r + n,c - n: c + n) == unique_v(i)));
        if count > 15
          R(r, c) = unique_v(i);
          break;
        endif
      endfor
    endfor
  endfor
  R = CutPadding(R, n);
endfunction


I = imread("Image25.jpg");
G = rgb2gray(im2double(I));
imshow(G);
SE = [0, 1, 1, 1, 0; 1, 2, 2, 2, 1; 1, 1, 5, 1, 1; 1, 2, 2, 2, 1; 0, 1, 1, 1, 0];
R = median(G, SE);
imshow(R);
