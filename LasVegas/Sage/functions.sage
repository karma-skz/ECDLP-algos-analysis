def create_P(F,P,noP):
	import random
        import numpy as np
	factor_base=np.empty(shape=(noP,1),dtype=object)
	ord=P.order()
        seq_m=np.empty(shape=(noP,1),dtype=object)
	x=0
	while x<noP:
		a=randrange(1,ord)
		if a not in seq_m:
			temp_R=a*P
			factor_base[x]=temp_R
                        a=mod(a,ord)
			seq_m[x]=a
                        x=x+1
	out=[factor_base,seq_m]
	return out
def create_Q(F,Q,noQ):
        import random
        import numpy as np
        factor_base=np.empty(shape=(noQ,1),dtype=object)
        ord=Q.order()
        seq_Q=np.empty((noQ,1),dtype=object)
        x=0
        while x<noQ:
                a=randrange(1,ord)
                if a not in seq_Q:	
			temp_R=-a*Q
			factor_base[x]=temp_R
                        a=mod(a,ord)
			seq_Q[x]=a
                        x=x+1
	out=[factor_base,seq_Q]
	return out
def create_kernel(F,P,Q,ex_noQ,ex_noP,n1):
        import numpy as np
        R.<x,y,z>=PolynomialRing(F,3,order='lex')
        degs = WeightedIntegerVectors(n1, [1,1,1])
        mons=[x^d[0] * y^d[1] * z^d[2] for d in degs]
        length=len(mons)
        mat_m=np.empty(shape=(3*n1+ex_noQ+ex_noP,length),dtype=object) ## look at the shape
        np_ker_matrix=np.empty(shape=(ex_noP+ex_noQ,3*n1+ex_noQ+ex_noP),dtype=object)
        out_P=create_P(F,P,3*n1+ex_noP-1)
        seq_P=out_P[0];seq_m=out_P[1]
        out_Q=create_Q(F,Q,ex_noQ)
        seq_Q=out_Q[0];seq_Q_r=out_Q[1]
        Q=-Q
        temp_row=np.array([mons[i](Q.element()[0],Q.element()[1],Q.element()[2]) for i in range(length)],dtype=object,copy=False)
        mat_m[0]=temp_row
        for j in range(ex_noQ):
                temp_row=np.array([mons[i](seq_Q[j][0].element()[0],seq_Q[j][0].element()[1],seq_Q[j][0].element()[2]) for i in range(length)],dtype=object,copy=False)
                mat_m[j+1]=temp_row
        for j in range(3*n1+ex_noP-1):
    	        temp_row=np.array([mons[i](seq_P[j][0].element()[0],seq_P[j][0].element()[1],seq_P[j][0].element()[2]) for i in range(length)],dtype=object,copy=False)
                mat_m[j+ex_noQ+1]=temp_row
        mat_m=matrix(F,mat_m)
        ker=mat_m.left_kernel(basis='pivot')
        ker=ker.basis()                      
        for i in range(ex_noP+ex_noQ):
                np_ker_matrix[i]=np.array(ker[i],dtype=object,copy=False)
        del ker
        out_return=np.array([np_ker_matrix,seq_m,seq_Q_r],dtype=object,copy=False)
        return out_return
def column_reduce(F,M,noCol,pivot,noRows):
    import numpy as np
    np.set_printoptions(linewidth=250)
    MM=M
    temp_element=MM[pivot,noCol]
    for i in range(pivot):
        temp_element1=MM[i,noCol]
        temp_element2=-temp_element1/temp_element
        MM[i,:]=MM[i,:]+temp_element2*MM[pivot,:]
    for i in range(pivot+1,noRows):
        temp_element1=MM[i,noCol]
        temp_element2=-temp_element1/temp_element
        MM[i,:]=MM[i,:]+temp_element2*MM[pivot,:]
    return MM
def scanMatrix(M,F,t):
        import numpy as np
	nrows=M.shape[0]
	ncols=M.shape[1]
	out=0;nzeros=-1
	for i in range(nrows):
		nzeros=np.count_nonzero(M[i,:]==F(0))
		if nzeros>=t:
			return out+1,i
	return out,nzeros
### Computes the DLP
def compute_exp(F,vec,seq_Q_r,seq_m,no_Q,no_P,ord):
        temp_sum_q=mod(0,ord);temp_sum_p=mod(0,ord)
        if vec[0]<>F(0):
	        temp_sum_q=temp_sum_q+mod(1,ord)
	for k in range(1,no_Q+1):
		if vec[k]<>F(0):
			temp_sum_q=temp_sum_q+seq_Q_r[k-1]
	for k in range(no_Q+1,len(vec)):
		if vec[k]<>F(0):
			temp_sum_p=temp_sum_p+seq_m[k-no_Q-1]
	dlp=(temp_sum_q)^(-1)*temp_sum_p
        return dlp
def create_cyclic_curve(F):
	temp_variable=0
	while temp_variable==0:
		a=F.random_element()
        	b=F.random_element()
        	E=EllipticCurve([a,b])
		G=E.abelian_group()
		if G.is_cyclic():
			temp_g=G.gens()[0].order()
			if is_prime(temp_g):
				temp_variable=1
	return G
