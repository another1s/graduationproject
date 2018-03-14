function x=duffing()
%to change the 'amp' value you can see the period doubling scenario
amp=0.42;  % control parameter
b=0.5; 
alpha=-1.0d0;
beta=1.0d0;
w=1.0;
%time step and initial condition          
tspan = 0:0.1:500;  
x10 = 0; x20 = -1.8;
y0 = [x10; x20];
op=odeset('abstol',1e-9,'reltol',1e-9);          
[t,y] = ode45(@(t,x) f(t,x,b,alpha,beta,amp,w),tspan,y0,op);
x1=y(:,1); x2=y(:,2);
x=[x1,x2];
%  plot(x1,x2);  %plot the variable x and y          
%  fprintf(fid,'%12.6f\n',x1,x2);
            
function dy = f(t,y,b,alpha,beta,amp,w)        
x1 = y(1);    x2 = y(2);  
dx1=x2;
dx2=-b*x2-alpha*x1-beta*x1^3+amp*sin(w*t);	
dy = [dx1; dx2];