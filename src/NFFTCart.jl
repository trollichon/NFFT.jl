module NFFTCart

using Cartesian
import Base.ind2sub
import Cartesian.@nloops
import Cartesian.@nexprs
import Cartesian.@nref

export * ,@nmutliply 
#,NFFTPlan, nfft, nfft_adjoint, ndft, ndft_adjoint, nfft_performance


# Some internal documentation (especially for people familiar with the nfft)
#
# - Currently the window cannot be changed and defaults to the kaiser-bessel
#   window. This is done for simplicity and due to the fact that the 
#   kaiser-bessel window usually outperforms any other window
#
# - The window is precomputed during construction of the NFFT plan
#   When performing the nfft convolution, the LUT of the window is used to
#   perform linear interpolation. This approach is reasonable fast and does not
#   require too much memory. There are, however alternatives known that are either 
#   faster or require no extra memory at all.
#
#



function window_kaiser_bessel(x,n,m,sigma)
  b = pi*(2-1/sigma)
  arg = m^2-n^2*x^2
  if(abs(x) < m/n)
    y = sinh(b*sqrt(arg))/sqrt(arg)/pi
  elseif(abs(x) > m/n)
    y = zero(x)
  else
    y = b/pi
  end
  return y
end

function window_kaiser_bessel_hat(k,n,m,sigma)
  b = pi*(2-1/sigma)
  return besseli(0,m*sqrt(b^2-(2*pi*k/n)^2))
end

type NFFTPlan{D,T}
  N::NTuple{D,Int}
  M::Int
  x::Array{T,2}
  m::Int
  sigma::T
  n::NTuple{D,Int}
  K::Int
  windowLUT::Vector{Vector{T}}
  windowHatInvLUT::Vector{Vector{T}}
  tmpVec::Array{Complex{T},D}
end

function NFFTPlan{D,T}(x::Array{T,2}, N::NTuple{D,Int}, m=4, sigma=2.0, K=2000)
  
  if D != size(x,1)
    throw(ArgumentError())
  end

      

  n = ntuple(D, d->int(round(sigma*N[d])) )

  tmpVec = zeros(Complex{T}, n)

  M = size(x,2)

  # Create lookup table
  
  windowLUT = Array(Vector{T},D)
  for d=1:D
    Z = int(3*K/2)
    windowLUT[d] = zeros(T, Z)
    for l=1:Z
      y = ((l-1) / (K-1)) * m/n[d]
      windowLUT[d][l] = window_kaiser_bessel(y, n[d], m, sigma)
    end
  end

  windowHatInvLUT = Array(Vector{T}, D)
  for d=1:D
    windowHatInvLUT[d] = zeros(T, N[d])
    for k=1:N[d]
      windowHatInvLUT[d][k] = 1. / window_kaiser_bessel_hat(k-1-N[d]/2, n[d], m, sigma)
    end
  end

  plan = NFFTPlan(N, M, x, m, sigma, n, K, windowLUT, windowHatInvLUT, tmpVec )
  
  # generate convolution and adjoint convolution functions according to dimension of problem
  myConv = initConv(D)
  myAdj  = initConvAdj(D)
  myApo = initApo(D)
  myApoAdj  = initApoAdj(D)
  
  return plan
  
end

function NFFTPlan{T}(x::Array{T,1}, N::Integer, m=4, sigma=2.0)
  NFFTPlan(reshape(x,1,length(x)), (N,), m, sigma)
end

### nfft functions ###

function nfft!{T,D}(p::NFFTPlan{D}, f::Array{T,D}, fHat::Vector{T})
  p.tmpVec[:] = 0
  @inbounds apodization!(p, f, p.tmpVec)
  fft!(p.tmpVec)
  @inbounds convolve!(p, p.tmpVec, fHat)
  return fHat
end

function nfft{T,D}(p::NFFTPlan, f::Array{T,D})
  fHat = zeros(T, p.M)
  nfft!(p, f, fHat)
  return fHat
end

function nfft{T,D}(x, f::Array{T,D})
  p = NFFTPlan(x, size(f) )
  return nfft(p, f)
end

function nfft_adjoint!{T,D}(p::NFFTPlan{D}, fHat::Vector{T}, f::Array{T,D})
  p.tmpVec[:] = 0
  @inbounds convolve_adjoint!(p, fHat, p.tmpVec)
  ifft!(p.tmpVec)
  p.tmpVec *= prod(p.n)
  @inbounds apodization_adjoint!(p, p.tmpVec, f)
  return f
end

function nfft_adjoint{T,D}(p::NFFTPlan{D}, fHat::Vector{Complex{T}})
  f = zeros(Complex{T},p.N)
  nfft_adjoint!(p, fHat, f)
  return f
end

function nfft_adjoint{T,D}(x, N::NTuple{D,Int}, fHat::Vector{T})
  p = NFFTPlan(x, N)
  return nfft_adjoint(p, fHat)
end

### ndft functions ###

# fallback for 1D 
function ind2sub{T}(::Array{T,1}, idx)
  idx
end

function ndft{T,D}(plan::NFFTPlan{D}, f::Array{T,D})
  g = zeros(T, plan.M)

  for l=1:prod(plan.N)
    idx = ind2sub(plan.N,l)
    for k=1:plan.M
      arg = zero(T)
      for d=1:D
	arg += plan.x[d,k] * ( idx[d] - 1 - plan.N[d] / 2 )
      end
      g[k] += f[l] * exp(-2*pi*1im*arg)
    end
  end

  return g
end

function ndft_adjoint{T,D}(plan::NFFTPlan{D}, fHat::Array{T,1})

  g = zeros(T, plan.N)
  for l=1:prod(plan.N)
    idx = ind2sub(plan.N,l)
    for k=1:plan.M
      arg = zero(T)
      for d=1:D
	arg += plan.x[d,k] * ( idx[d] - 1 - plan.N[d] / 2 )
      end
      g[l] += fHat[k] * exp(2*pi*1im*arg)
    end
  end

  return g
end



### convolve! ###


# helper function
function genCodeAndGenerateFunction(functionExpression::Expr)
   #
   #Return as tuple: 1. an object to the top level generateed function
   #		     2. the function code after macroexpand magic
   #		     3. the function code before  macroexpand magic
  
   functionCode = macroexpand(functionExpression)
   fun = eval(functionExpression)
   return (fun,functionCode,functionExpression)
end


function initConvAdj(D)      
	funcExprDepD=quote 
	    function convolve_adjoint!{T}(p::NFFTCart.NFFTPlan{$D}, fHat::Array{T,1}, g::Array{T,$D})
	    scale = 1.0 / p.m * (p.K-1);
	    @Cartesian.nexprs $D (d->n_d=p.n[d]) ; # generate local variables n1=p.n[1], n2=p.n[2], ..., nD=p.n[D]
	    for k=1:p.M  # loop over non equispaced nodes
		@Cartesian.nexprs 1 d->(tmpQ_{d+($(D))}=fHat[k] );
		@Cartesian.nexprs $D (d->c_d=int((p.x[d,k]*n_d))) # generate local variables c1, c2, ...cD
		#= Cartesian style loop over non zero elements=#
		@Cartesian.nloops $D l (l-> c_l-p.m : c_l+p.m) (
			d->(nothing;
			    idx_d=((l_d+n_d)%n_d)+1;
			    idxb=abs((p.x[d,k]*n_d - l_d)*scale) + 1;
			    idxbL=int((idxb) );
			    tmpQ_d = tmpQ_{d+1}*(p.windowLUT[d][idxbL] + ( idxb-idxbL ) * (p.windowLUT[d][idxbL+1] - p.windowLUT[d][idxbL] ) );
			    )
			    )  begin
				  @Cartesian.nexprs 1 d->((@nref $D g idx) +=tmpQ_d )
			      end 
		# end of nested loops code generation
		end # of for k=1:p.M  
	end # of function
    end # of quote 
	    
    return genCodeAndGenerateFunction(funcExprDepD)

end


function initConv(D)
	funcExprDepD=quote 
	function convolve!{T}(p::NFFTPlan{$D}, g::Array{T,$D}, fHat::Array{T,1})
	    scale = 1.0 / p.m * (p.K-1);
	    @Cartesian.nexprs $D (d->n_d=p.n[d]) ; # generate local variables n1=p.n[1], n2=p.n[2], ..., nD=p.n[D]
	    for k=1:p.M  
		@Cartesian.nexprs 1 d->(tmpQ_{d+($(D))}=1.0 );
  	        @Cartesian.nexprs $D (d->c_d=int(floor(p.x[d,k]*n_d))) # generate local variables c1, c2, ...cD
	        @Cartesian.nloops $D l (l-> c_l-p.m : c_l+p.m) (
			d->(idx_d=((l_d+n_d)%n_d)+1;
			    idxb=abs((p.x[d,k]*n_d - l_d)*scale) + 1;
			    idxbL=int(floor(idxb) );
			    tmpQ_d = tmpQ_{d+1}*(p.windowLUT[d][idxbL] + ( idxb-idxbL ) * (p.windowLUT[d][idxbL+1] - p.windowLUT[d][idxbL] ) ))
			    )  begin
				  @Cartesian.nexprs 1 d->(fHat[k] +=tmpQ_d * (@Cartesian.nref $D g idx) )
			      end
		# end of nested loops code generation
	    end # of for k=1:p.M  
	end # of function  
    end # of quote
    return genCodeAndGenerateFunction(funcExprDepD)
end

### apodization! ###
# poor hacking of additional Cartesian goodies
# maybe it works directly with some Cartesians tools.
#  Macro @nmutliply takes as arguments:
#     - D: number of objects
#     - myExp : an expression which will be passed to Cartesian@ntuple call
#                examples : myExp= :(A[d]) or myExp=:(A[d]*l_d+1) 
#    Example:
#    @nmutliply 3 :(A[d])  -returns-> :( A[1]*A[2]*A[3]) 


*(a::Expr,b::Expr)=:($a * $b)

macro nmutliply(D,myExp)
    #t= macroexpand(:(@Cartesian.ntuple 2 d->A[d][l_d]))
    if D==1
	t=macroexpand(:(@Cartesian.ntuple $D d->$myExp))
	# get a tuple of args ? 
	z=(t.args)
	res = z[1]
    else
	t=macroexpand(:(@Cartesian.ntuple $D d->$myExp))
	z=(t.args)
	res=*(z...) # multiplication of Expressions !
    end
    return res
end

function initApo(D)
    prodWinExpr=eval(quote @nmutliply $D :(p.windowHatInvLUT[d][l_d]) end)
    funcExprDepD=quote
        function apodization!{T}(p::NFFTCart.NFFTPlan{$D}, f::Array{T,$D}, g::Array{T,$D})
	    @Cartesian.nexprs $D d->n_d=p.n[d];
	    @Cartesian.nexprs $D d->N_d=p.N[d];         
	    @Cartesian.nexprs $D d->const offset_d = int( n_d - N_d / 2 ) - 1;                           
	    @Cartesian.nloops $D l (d->1:N_d) (d->(
		begin
			idx_d = ((l_d+offset_d)% n_d) + 1
		end)
	    ) begin
		(@Cartesian.nref $D g idx) = (@Cartesian.nref $D f l)  *$(prodWinExpr)
	    end
	end 
    end # of quote
   return genCodeAndGenerateFunction(funcExprDepD)
end

### apodization_adjoint! ###
function initApoAdj(D)
  prodWinExpr=eval(quote @nmutliply $D :(p.windowHatInvLUT[d][l_d]) end)
  funcExprDepD=quote    	  
      function apodization_adjoint!{T}(p::NFFTCart.NFFTPlan{$D}, g::Array{T,$D}, f::Array{T,$D})
            @Cartesian.nexprs $D d->n_d=p.n[d];
	    @Cartesian.nexprs $D d->N_d=p.N[d];         
	    @Cartesian.nexprs $D d->const offset_d = int( n_d - N_d / 2 ) - 1;                           	    
	    @Cartesian.nloops $D l (d->1:N_d) (d->(
	    begin
		idx_d = ((l_d+offset_d)% n_d) + 1
		#tmpWin_d*=p.windowHatInvLUT[d][l_d]
	    end)
	    ) begin
	      (@Cartesian.nref $D f l) = (@Cartesian.nref $D g idx) * $(prodWinExpr);
	    end
    end
  end 
  return genCodeAndGenerateFunction(funcExprDepD)
end


### performance test ###

function nfft_performance()

  m = 4
  sigma = 2.0

  # 1D

  N = 2^19
  M = N

  x = rand(M) .- 0.5
  fHat = rand(M)*1im

  println("NFFT Performance Test 1D")

  tic()
  p = NFFTPlan(x,N,m,sigma)
  println("initialization")
  toc()

  tic()
  fApprox = nfft_adjoint(p,fHat)
  println("adjoint")
  toc()

  tic()
  fHat2 = nfft(p, fApprox);
  println("trafo")
  toc()

  N = 1024
  M = N*N

  x2 = rand(2,M) .- 0.5
  fHat = rand(M)*1im

  println("NFFT Performance Test 2D")

  tic()
  p = NFFTPlan(x2,(N,N),m,sigma)
  println("initialization")
  toc()

  tic()
  fApprox = nfft_adjoint(p,fHat)
  println("adjoint")
  toc()

  tic()
  fHat2 = nfft(p, fApprox);
  println("trafo")
  toc()
  
  N = 32
  M = N*N*N

  x3 = rand(3,M) .- 0.5
  fHat = rand(M)*1im

  println("NFFT Performance Test 3D")

  tic()
  p = NFFTPlan(x3,(N,N,N),m,sigma)
  println("initialization")
  toc()

  tic()
  fApprox = nfft_adjoint(p,fHat)
  println("adjoint")
  toc()

  tic()
  fHat2 = nfft(p, fApprox);
  println("trafo")
  toc()  

end



end
