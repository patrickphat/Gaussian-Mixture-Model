class GaussianMixtureModel:
    
    # Notation
    # dim: dimension of the data
    # cov: covariance matrix
    # n_clusters: number of clusters
    # N: normal distribution
    # m: number of data points
    # k: k-th cluster
    # i: i-th sample
    
    def __init__(self,n_clusters=5):
        
        # Number of clusters that the model converge into
        self.n_clusters = n_clusters
        
        # Vectorization of different guassian distribution
        # Each line is a distribution parameterized by mu and sigma
        self.gaussians = {}
        
        # init mu, shape = (K x dim)
        self.gaussians['mu'] = None
        
        # init cov matrix, shape = (K x dim x dim)
        self.gaussians['cov'] = None
        
        # init prior (sum to 1), shape = (1,K)
        self.gaussians['prior'] = None
        
        # dimension of data
        self.dim = None
        
        # matrix of likelihood of data point to each cluster, shape (m,K)
        self.likelihood = None
        
        # matrix of normal prob of each data point, shape (m,K)
        self.normal_matrix = None
        
        # abnormal threshold
        self.abnormal_threshold = None
        
        # save history
        self.history = {}
        self.history['log_likelihood'] = []
        
        
        
    def createCov(self):
        # Create a random covaraince matrix that positive definite
        
        while True:
            A = np.random.normal(size=(self.dim,1))
            B= A*A.T
            # Make sure the matrix is invertible and positive definite
            if np.isfinite(np.linalg.cond(B)) and np.linalg.det(B)>0:
                break
            else:
                continue
        return B
    
   
    def normal(self,X,k): 
        # Given the list of observation X is from a cluster, what's the likelihood of seeing X.
        # Return shape: (m,1)
        # Get dimension of data
        dim = X.shape[1]
        
        # retrieve mu and covariance matrix from cluster k
        mu_k = self.gaussians['mu'][k]
        cov_k = self.gaussians['cov'][k]

        var = multivariate_normal(mean=mu_k, cov=cov_k,allow_singular=True)
        result = np.expand_dims(var.pdf(X),1)
        return  result

    
    
    
    def proba(self,X):
        # The probability of observing x_i in the data. 
        # Return shape: (m,1)
        
        # Get number of data
        m = X.shape[0]
        
        # Retrieve prior
        prior = self.gaussians['prior']
        
        # Get dimension of data
        dim = X.shape[1]
        
        # initialize array of size (m,k), each line represents normal(x_i,k) for each cluster k
        normal_matrix = np.empty((m,0))
        
        # for each cluster, append the N result to the right of the matrix
        for k in range(self.n_clusters):
            normal_k = self.normal(X,k)
            normal_matrix = np.hstack((normal_matrix,normal_k))
        self.normal_matrix = normal_matrix
        
        # times with prior for each cluster then sum for each line
        proba = np.sum(normal_matrix*prior,axis=1)
        return np.expand_dims(proba,1)
        
    def step(self,X):
        
        # Get number of data
        m = X.shape[0]
        
        # Get dimension of data
        dim = X.shape[1]
        
        p = self.proba(X)
        # Matrix of likelihood. Shape of (m,k)
        likelihood = np.empty((m,0))
        
        # EXPECTATION
        for k in range(self.n_clusters):
            # retrieve mu, prior and covariance matrix
            mu_k = self.gaussians['mu'][k]
            cov_k = self.gaussians['cov'][k]
            prior_k = self.gaussians['prior'][0][k]
           
            
            # compute normal N(x|mu,Sigma)
            
            N_k = self.normal(X,k)
            # compute likehood at k-th
            lh_k = N_k*prior_k/p
            # likelihood, append to the right
            likelihood = np.hstack((likelihood,lh_k))
            
        
        self.likelihood = likelihood
    
        # MAXIMIZATION

        # Sum of all likelihood allocated to cluster k. Shape (1,k)
        m_k = np.sum(likelihood,axis=0)
        # Update prior and expand dim to (1,K)
        self.gaussians['prior'] = np.expand_dims(m_k/m,axis=0)

        # Update mean
        for k in range(self.n_clusters):
            # likelihood at k-th. shape (m,1)
            lh_k = np.expand_dims(likelihood[:,k],1)
            self.gaussians['mu'][k] = 1/m_k[k]*np.sum(X*lh_k,axis=0)
              
        # Update cov
        for k in range(self.n_clusters):
            
            # retrieve mu_k
            mu_k = self.gaussians['mu'][k]
            cov = np.zeros((dim,dim))
            
            for i in range(m):
                
                diff = np.expand_dims((X[i]-mu_k),0)
                cov += likelihood[i][k]*(diff.T.dot(diff))/m_k[k]
            cov_k=cov

        
            self.gaussians['cov'][k] = cov_k
            
        
    
    def log_likelihood(self,X):
        
        m = X.shape[0]

        # Retrieve prior
        prior = self.gaussians['prior']
    
        normal_matrix = np.empty((m,0))
        
        # for each cluster, append the N result to the right of the matrix
        for k in range(self.n_clusters):
            normal_k = self.normal(X,k)
            normal_matrix = np.hstack((normal_matrix,normal_k))

        return np.log((normal_matrix*prior).sum(axis=1)).sum()
    
    def fit(self,X, max_iters=15):
        
        m = X.shape[0]

        # Get dim of data
        self.dim = X.shape[1]
        
        # init mu, shape = (K x dim)
        mulist=[]
        for k in range(self.n_clusters):
            random_mu = X[np.random.randint(low=0,high=len(X))]
            mulist.append(random_mu)
        self.gaussians['mu'] = np.array(mulist)

        # init covariance matrixes based on different sample of data, shape = (K x dim x dim)
        # Make sure every k-th covariance has det > 0
        self.gaussians['cov'] = np.array([self.createCov() for i in range(self.n_clusters)])

        
        # init prior (sum to 1), shape = (1,K)
        self.gaussians['prior']=np.random.dirichlet(np.ones(self.n_clusters),size=1)
                              
        
        for i in range(max_iters):
            print("step ",i,end=' ')
            # EM 
            self.step(X)
            
            # calculate log likelihood
            log_lh = self.log_likelihood(X)
            
            
            self.history['log_likelihood'].append(log_lh)
            print("log likelihood: ",log_lh)
            
    def fit_anomaly(self,X,y):
        
        # Get the probabilities of X, shape (m,1)
        proba = self.proba(X) 
        
        best_thresh = 0
        best_f1_score = 0
        for thresh in np.arange(0.0, 1.0, 0.01):
            y_pred = (proba < thresh)*1
            current_f1_score = f1_score(y,y_pred)
            if current_f1_score > best_f1_score:
                best_thresh = thresh
                best_f1_score = current_f1_score
        self.abnormal_thresh = best_thresh
        
    def predict_anomaly(self,X):
        return ((self.proba(X) < self.abnormal_thresh)*1)[:,0]
        
    def predict(self,X):
        return self.likelihood.argmax(axis=1)