
I = imread("fence.png");
G = rgb2gray(I);
#imwrite(G, "Results/Bai02/GrayImage.png")
W = zeros(101);
W(:, 51) = ones(101, 1);
W(51, :) = ones(1, 101);
