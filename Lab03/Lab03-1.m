clear
pkg load image

I1 = imread("barCodesDetection.png")
IG = rgb2gray(I1)
imshow(IG)
idx = (IG(:, :) <= 120)
con = zeros(size(idx))
IB = imerode(idx, SE)

function OneComponentAtATime(Image)
  [M, N]=size(Image);
  Connected = zeros(M,N);
  Mark = Value;
  Difference = Increment;
  Offsets = [-1; M; 1; -M];
  Index = [];
  No_of_Objects = 0;

 for i: 1:M :
     for j: 1:N:
          if(Image(i,j)==1)
               No_of_Objects = No_of_Objects +1;
               Index = [((j-1)*M + i)];
               Connected(Index)=Mark;
               while ~isempty(Index)
                    Image(Index)=0;
                    Neighbors = bsxfun(@plus, Index, Offsets');
                    Neighbors = unique(Neighbors(:));
                    Index = Neighbors(find(Image(Neighbors)));
                    Connected(Index)=Mark;
               end
               Mark = Mark + Difference;
          end
    end
end
