clear all

%% SET OUTPUT PATH
pathMain1= 'data\RLN\SimuMix3D_128e\gt\';
% pathMain1= 'data\RLN\SimuMix3D_256\gt\';
% pathMain1= 'data\RLN\SimuMix3D_382\gt\';
mkdir(pathMain1)

% How many simulation to generate
number_of_simulation=20;

%%
% WITH BACKGROUND, for applying simulated data to biology sample
% you can choose True
is_with_background=false;

Rsphere = 9;
Ldot = 9;

% set the Gaussian blur parameter
delta=0.7;

% Set numbers of strctures
% n_spheres=100;
% n_ellipsoidal=100;
% n_dots=400/8;

n_spheres=1600;
n_ellipsoidal=1600;
n_dots=800;

%% creat Gaussian filter
Ggrid = -floor(5/2):floor(5/2);
[X, Y, Z] = meshgrid(Ggrid, Ggrid, Ggrid);

% Create Gaussian Mask
GaussM = exp(-(X.^2 + Y.^2 + Z.^2) / (2*delta^2));

% Normalize so that total area (sum of all weights) is 1
GaussM = GaussM/sum(GaussM(:));

%%   spheroid
for tt=1:number_of_simulation
    disp(tt)

    A=zeros(256,256,256);
    % A=zeros(512,512,128);
    % A=zeros(128,128,128);
    % A=zeros(382,382,151);
    [Sx, Sy, Sz] = size(A);
    
    rrange = fix(Rsphere/2);
    xrange = Sx - 2*Rsphere;
    yrange = Sy - 2*Rsphere;
    zrange = Sz - 2*Rsphere;

    for times=1:n_spheres
        x=floor(xrange*rand()+Rsphere);
        y=floor(yrange*rand()+Rsphere);
        z=floor(zrange*rand()+Rsphere);
        
        r=floor(rrange*rand()+rrange);
        
        inten=800*rand()+50;
        
        
        for i=(x-r):(x+r)
            for j=(y-r):(y+r)
                for k=(z-r):(z+r)
                    
                    if(((i-x)^2+(j-y)^2+(k-z)^2)<=(r)^2)
                        A(i,j,k)=inten;
                    end
                end
            end
        end
    end
    
    for times=1:n_ellipsoidal
        x=floor(xrange*rand()+Rsphere);
        y=floor(yrange*rand()+Rsphere);
        z=floor(zrange*rand()+Rsphere);
        
        r1=floor(rrange*rand()+rrange);
        r2=floor(rrange*rand()+rrange);
        r3=floor(rrange*rand()+rrange);
        
        inten=800*rand()+50;
        
        for i=(x-r1):(x+r1)
            for j=(y-r2):(y+r2)
                for k=(z-r3):(z+r3)
                    if((((i-x)^2)/r1^2+((j-y)^2)/r2^2+((k-z)^2)/r3^2)<=1.3 && (((i-x)^2)/r1^2+((j-y)^2)/r2^2+((k-z)^2)/r3^2)>=0.8)
                        A(i,j,k)=inten;
                    end
                end
            end
        end
    end
    
    dotrangex = Sx - Ldot -1;
    dotrangey = Sy - Ldot -1;
    dotrangez = Sz - Ldot -1;

    for times=1:n_dots
        x=floor((Sx-3)*rand()+1);
        y=floor((Sy-3)*rand()+1);
        z=floor((Sz-3)*rand()+1);
        
        r=1;
        
        inten=800*rand()+50;
        
        A(x:x+1,y:y+1,z:z+1)=inten;
        
    end
    
    for times=1:n_dots
        x=floor(dotrangex*rand()+1);
        y=floor((Sy-3)*rand()+1);
        z=floor((Sz-3)*rand()+1);
        
        r=1;
        
        inten=800*rand()+50;
        k=floor(rand()*Ldot)+1;
        
        A(x:x+k,y:y+1,z:z+1)=inten;
    end
    
    for times=1:n_dots
        x=floor((Sx-3)*rand()+1);
        y=floor(dotrangey*rand()+1);
        z=floor((Sz-3)*rand()+1);
        
        r=1;
        
        inten=800*rand()+50;
        
        k=floor(rand()*9)+1;
        
        A(x:x+1,y:y+k,z:z+1)=inten+50*rand();
    end
    
    for times=1:n_dots
        x=floor((Sx-3)*rand()+1);
        y=floor((Sy-3)*rand()+1);
        z=floor(dotrangez*rand()+1);
        
        r=1;
        
        inten=800*rand()+50;
        k=floor(rand()*Ldot)+1;
        
        A(x:x+1,y:y+1,z:z+k)=inten;
        
    end
    
    for times=1:n_dots
        x=floor(dotrangex*rand()+1);
        y=floor((Sy-3)*rand()+1);
        z=floor(dotrangez*rand()+1);
        
        r=1;
        
        inten=800*rand()+50;
        k1=floor(rand()*Ldot)+1;
        k2=floor(rand()*Ldot)+1;
        
        A(x:x+k1,y:y+1,z:z+k2)=inten;
        
    end
    
    for times=1:n_dots
        x=floor(dotrangex*rand()+1);
        y=floor(dotrangey*rand()+1);
        z=floor((Sz-3)*rand()+1);
        
        r=1;
        
        inten=800*rand()+50;
        
        k1=floor(rand()*Ldot)+1;
        k2=floor(rand()*Ldot)+1;
        A(x:x+k1,y:y+k2,z:z+1)=inten;
        
    end
    
    for times=1:n_dots
        x=floor((Sx-3)*rand()+1);
        y=floor(dotrangey*rand()+1);
        z=floor(dotrangez*rand()+1);
        
        r=1;
        
        inten=800*rand()+50;
        k1=floor(rand()*Ldot)+1;
        k2=floor(rand()*Ldot)+1;
        A(x:x+1,y:y+k1,z:z+k2)=inten;
        
    end
    
    for times=1:n_dots
        x=floor(dotrangex*rand()+1);
        y=floor(dotrangey*rand()+1);
        z=floor(dotrangez*rand()+1);
        
        r=1;
        
        inten=800*rand()+50;
        k1=floor(rand()*Ldot)+1;
        k2=floor(rand()*Ldot)+1;
        k3=floor(rand()*Ldot)+1;
        
        A(x:x+k1,y:y+k2,z:z+k3)=inten;
        
    end
    
    %WriteTifStack(A, [pathMain1,num2str(tt),'.tif'], 32);
    if is_with_background
        A=A+30;
    end
    
    
    A = convn(A, GaussM, 'same');
    
    WriteTifStack(A, [pathMain1,num2str(tt),'.tif'], 32);
end


