%-------------------
% N is the number of vertices or neurons

function f=RBF()
inputNums=2; % input x1,x2,x3  
outputNums=2; % output x1_,x2_,x3_ 
hideNums=121; % neurons number
maxcount=1000; % calculator
samplenum=2500; % train sample number  
precision=0.001; %  approximation rate

%neurons distributed among 6*6
w=rand(11,11); %w is a 6*6 weight matrix of neurons 

% training stage
% in the first 200 seconds, the weight will be updated through kr
% f denotes the approximation of the status
t0=1;


a=linspace(-3,3,11);
a=repmat(a,1,11);
a=sort(a);
b=linspace(-3,3,11);
b=repmat(b,1,11);
%center should be 2*121 ,which is the locations of the neurons
center=[a;b]';

theta=ones(11,11)*0.001;
% x is the trajectory of duffing function for 500 seconds
% using the first 1000 points for training 
x=duffing();
while(t0<200)
    v=reshape(x(t0,:),1,2);
    f=w'*gauss(v,center)+theta;
    t0=t0+1;
    plot(t0,f)
    
end 

end
% w stands for weight martrix 
% zeta=2 sigma=0.001
% v stands for the location of the estimate of the points 
function [w]=weigh(w,v,c)
zeta=2;
dt=0.1;
sigma=0.001;
dw=(-1)*zeta*gauss(v,x)*v-sigma*zeta*w;
w=w+dw*dt;
end

% v symbolizes vertices ,which is the estimate of the points
% v should be 1*2, c is a 121*2
% c is the center of neurons

function [s]=gauss(v,c)
v=repmat(v,121,1);
A=[];
for i=1:121
    a1=v(i,:);
    b1=c(i,:);
    m=(a1-b1)';
    t=exp(((-1)*m)'*m/0.25);
    A(i)=t;
end
s=reshape(A,11,11);
end 