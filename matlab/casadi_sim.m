function [ output_args ] = casadi_sim( input_args )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here


% low-level operations
symb_matrix = SX. sym ( 'Z ' , 4 , 2 ) ;
dense_matrix = SX.zeros(4,5);
sparse_matrix = SX(4,5);

C = DMatrix ( 2 , 3 );
% matlab functions
C_sparse = sparse (C);
C_dense = full(C);

% left hand-side
x = MX. sym ( 'x ' , 2 , 2 ) ;
y = MX. sym ( 'y ' ) ;
f = 3*x + y ;

% sparse matrix CCS format

a = Sparsity.upper(5);
disp(SX.sym('x',a));


M = diag (SX ( [ 3 , 4 , 5 , 6 ] ) ) ;
disp (M( 2 : end , 2 : 2 : 4 ));

%concat
y = SX. sym ( 'y ' ,5) ;
x = SX. sym ( 'x ' ,5) ;
[x,y];
[x;y];

%split
x = SX. sym ( 'x' , 5 , 2 ) ;
w = horzsplit(x,1);     % significantly more efficient for MX
w = {x(1:3,:),x( 4:5,:)}; %alternative

% linear algebra
A = SX. sym ( 'A ' , 3 , 3 ) ;
x = SX. sym ( ' x ' , 3 ) ;
jacobian(A*x^2,x)


%functions 
f = SXFunction ( ' f ' ,{x ,y },{x , sin(y)*x } ) ;

x = SX. sym ( ' x ' ) ;
p = SX. sym ( ' p ' ) ;
f = x^2
g = log(x) - p


nlp = SXFunction ( ' nlp ' , ...
nlpIn ( 'x' , x , 'p' , p ) , ...
nlpOut ( 'f' , f , 'g' , g ) ) ;
nlp
nlp.jacobian(0,1)
nlp.jacobian('x','g')
res = nlp ( { 1 , 1 } )
res_str = nlp (struct('x',  1 , 'p', 1  ))
res_str.f
res_str.g

op = struct;
op.input_scheme=char( 'x' , 'p' ) ;
op.output_scheme=char( 'f' , 'g' ) ;
nlp = SXFunction ( 'nlp' , ...
{x , p} ,{ f , g } , op ) ;


sx_func = SXFunction ( nlp ); % speed up/ mem overhead


end


