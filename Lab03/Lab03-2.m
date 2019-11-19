pkg load image

%{
  Bai tap Lab-03 bai 2
%}

I = imread("fence.png");
G = imread("noiseSP.png");

# Cau 1
%{
  subplot(1, 2, 1);
  imshow(I);
  subplot(1, 2, 2);
  imshow(G);
%}
# Het Cau 1

# Cau 2
N = I + G;
imwrite(N, "Results/N.png");
imshow(N);
# Het Cau 2
