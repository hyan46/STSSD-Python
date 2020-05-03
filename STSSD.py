import scipy.interpolate as spl
import numpy as np

def bsplineBasis(n, k,deg):
    """B-spline type matrix for splines.

    Returns a matrix whose columns are the values of the b-splines of deg
    `deg` as sociated with the knot sequence `knots` evaluated at the points
    `x`.

    Parameters
    ----------
    x : array_like
        Points at which to evaluate the b-splines.
    deg : int
        Degree of the splines.
    knots : array_like
        List of knots. The convention here is that the interior knots have
        been extended at both ends by ``deg + 1`` extra knots.

    Returns
    -------
    vander : ndarray
        Vandermonde like matrix of shape (m,n), where ``m = len(x)`` and
        ``m = len(knots) - deg - 1``

    Notes
    -----
    The knots exending the interior points are usually taken to be the same
    as the endpoints of the interval on which the spline will be evaluated.

    """
    knots = np.r_[np.zeros(deg),np.linspace(0,n-1,k),(n-1) * np.ones(deg)]
    x = np.arange(n)
    m = len(knots) - deg - 1
    v = np.zeros((m, len(x)))
    d = np.eye(m, len(knots))
    for i in range(m):
        v[i] = spl.splev(x, (knots, d[i], deg))
    return v.T

def chartOC( T2tr,mT2,sd):
    T2c = T2tr - mT2
    T2tn = T2c/sd
    Itr = np.argmax(T2tn,axis=0);
    Ttr = np.max(T2tn,0)
    return Ttr,Itr

def chartIC( T2tr,ifIncludeNAN = True, ndel = 0):
    T2trd= T2tr[:,ndel:];
    if ifIncludeNAN:
        mT2 = np.nanmean(T2trd,axis=1);
        sd = np.nanstd(T2trd,axis=1);
    else:
        mT2 = np.mean(T2trd,axis=1);
        sd = np.std(T2trd,axis=1);
    mT2 = mT2.reshape(-1,1)
    sd = sd.reshape(-1,1)
    T2c = T2trd - mT2
    T2tn = T2c/sd
    Itr = np.argmax(T2tn,axis=0);
    Ttr = np.max(T2tn,0)    
    return mT2,sd,Ttr,Itr



def ewmamonit( Y,B,Bs,lambda1,allgamma, isewma=False, maxIter = 2, L = 0, mT2 = 0, sd = 1, initial = 0):
    lambdat = lambda1[0]
    lambdaxy = lambda1[1:]
    lambdat1 = lambdat;
    sizey = Y.shape;
    ndim = len(sizey);
    issave = 1

    # define additional functions
    vec = lambda x: x.reshape(-1)
    gammalength = len(allgamma)
    softthreshold = lambda residual,gamma : np.sign(residual)*np.maximum(np.abs(residual) - gamma, 0)
    B.insert(0,[])
    constructD =lambda n: np.diff(np.eye(n),1,axis=0);
    nDim = len(Y.shape)
    nT = Y.shape[0]
    if issave:    
        thetaall = np.zeros(Y.shape);
        Sall = np.zeros(Y.shape);

    if Bs is None:
        LL = 2
        BetaS = Y*0
    elif len(Bs) == 1:
        LL =  2*np.linalg.norm(Bs[0],ord=2)**2
        X =  np.zeros(Bs[0].shape[1])
        BetaS = np.zeros(Bs[0].shape[1])
    elif len(Bs) == 2:
        LL = 2*np.linalg.norm(Bs[0],ord=2)**2*np.linalg.norm(Bs[1],ord=2)**2
        X = np.zeros((Bs[0].shape[1],Bs[1].shape[1]));
        BetaS = np.zeros((Bs[0].shape[1],Bs[1].shape[1]));

    D = [[] for i in range(ndim)];
    K = [[] for i in range(ndim)];
    H = [[] for i in range(ndim)];

    for idim in range(1,ndim):
        if B[idim] is None:
            B[idim] = np.eye(Y.shape[idim]);
        D[idim] = constructD(B[idim].shape[1]);
        H[idim] = B[idim]@ np.linalg.solve(B[idim].T@B[idim] + lambdaxy[idim-1] * (D[idim].T@D[idim]),B[idim].T);
    
    T2 = np.zeros((gammalength,nT))
    Tte = np.zeros(nT);
    Itr = np.zeros(nT,dtype=int);
    defect=0;
    tnew = 1;

    for t in range(nT):
        dall = [[] for i in range(gammalength)]
        thetai = [[] for i in range(gammalength)]
        e = np.zeros(gammalength);
        for i in range(gammalength):
            y = Y[t]
            if t==0:
                if ndim == 2:
                    Yhat = H[1]@Y[0]
                elif ndim == 3:
                    Yhat = H[1]@Y[0]@H[2];
            else:
                Snow = 0;
                Snowold = 1;
                iiter = 0;
                thetaold = Yhat;
                while iiter < maxIter:
                    iiter = iiter +1;
                    Snowold = Snow;
                    BetaSold = BetaS;
                    told = tnew;
                    if not isewma:
                        if ndim == 2:
                            yhat = H[1]@(y-Snow);
                        elif ndim == 3:
                            yhat = H[1]@(y-Snow)@H[2];
                        Yhat = lambdat1*yhat+(1-lambdat1)*thetaold;
                    else:
                        Yhat = lambdat1*(y-Snow)+(1-lambdat1)*thetaold;

                    if Bs is None:
                        residual = y - Yhat
                        Snow = softthreshold(residual, allgamma[i])
                    elif ndim ==2:
                        BetaSe = X + 2/LL* Bs[0].T@(y -Bs[0]@X - Yhat)
                        BetaS = softthreshold(BetaSe,allgamma[i]/LL); 
                        Snow = Bs[0] @BetaS;
                    elif ndim==3:
                        BetaSe = X + 2/LL* Bs[0].T@(y -Bs[0]@X@Bs[1].T - Yhat)@Bs[1];
                        BetaS = softthreshold(BetaSe,allgamma[i]/LL); 
                        Snow = Bs[0] @BetaS@ Bs[1].T;
                        tnew = (1+np.sqrt(1+4*told**2))/2;
                        if iiter==1:
                            X = BetaS
                        else:
                            X = BetaS+(told-1)/tnew*(BetaS-BetaSold);

                            
            if Bs is None:
                Ye = (y - Yhat)
                maxYe = np.max(np.abs(Ye.reshape(-1)));
                d = softthreshold(Ye,allgamma[i]);
            else:
                BetaSe = BetaS + 2/LL*Bs[0].T@(y -Bs[0]@BetaS@Bs[1].T - Yhat)@Bs[1];
                BetaS = softthreshold(BetaSe,allgamma[i]/LL); 
                Snow = Bs[0] @BetaS @ Bs[1].T
                d = Snow; 
            

            T2[i,t] = (np.sum(d*(y - Yhat)))**2/np.sum(d**2);
            dall[i] = d
            thetai[i] = Yhat;
        

        if L:
            L1,Itr[t] = chartOC( T2[:,t],mT2,sd);
            Tte[t] = L1;
            if L1 > L and t>initial:
                defect = dall[Itr[t]];
                break;
        else:
            if t < (initial+1):
                idx = np.argmax(T2[:,t])
                val = T2[idx,t]
                thetaall[t] = thetai[idx];
            else:
                Tte[t],Itr[t] = chartOC( T2[:,t],mT2,sd);
                Sall[t] = dall[Itr[t]]
                thetaall[t] = thetai[Itr[t]]



    if issave==1:
        Snow = Sall;
        Yhat = thetaall;
    elif issave == 2:
        Snow = dall;
        Yhat = thetai;

    return T2,Snow,Yhat,t,Itr,defect,Tte

