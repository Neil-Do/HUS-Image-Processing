%{
  Bai 2 cua bai thuc hanh Lab 2
%}



pkg load image

function K = CombineImages(I, J)
  k = 120;
  T = I > k;
  M = T(:, :, 1) & T(:, :, 2) & T(:, :, 3);
  N = ~M;
  M = uint8(M);
  N = uint8(N);
  K = M .* J + N .* I;
endfunction


# Cau 2-1

I = imread("foreground.png");
J = imread("background.png");

[Ir, Ic, Id] = size(I);
[Jr, Jc, Jd] = size(J);


J_resize = imresize(J, [Ir, Ic]);
imwrite(J_resize, "Result/backgroundResize.png");

%{

subplot(1, 3, 1);
imshow(J);
subplot(1, 3, 2);
imshow(J_resize);
subplot(1, 3, 3);
imshow(I);

%}
# Het Cau 2-1



# Cau 2-2
%{

I_reshape = reshape(I, Ir * Ic, 3);
hist_I = zeros(256, 3);

for i = 1:256
  hist_I(i, :) = sum(I_reshape == (i - 1));
endfor
plot(hist_I);

%}
# Het Cau 2-2




# Cau 2-3
K = CombineImages(I, J_resize);
imwrite(K, "Result/CombineImage.png");
imshow(K);
# Het Cau 2-3
