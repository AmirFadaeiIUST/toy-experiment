import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- math utils ----------------
def skew(v):
    x, y, z = v
    return np.array([[0,-z,y],[z,0,-x],[-y,x,0]], dtype=float)

def so3_exp(w):
    th = np.linalg.norm(w)
    I = np.eye(3)
    if th < 1e-12:
        return I + skew(w)
    K = skew(w/th)
    return I + np.sin(th)*K + (1-np.cos(th))*(K@K)

def se3_exp(xi):
    v, w = xi[:3], xi[3:]
    R = so3_exp(w)
    th = np.linalg.norm(w)
    I = np.eye(3)
    if th < 1e-12:
        V = I + 0.5*skew(w)
    else:
        K = skew(w/th)
        V = I + ((1-np.cos(th))/th)*K + ((th-np.sin(th))/th)*(K@K)
    t = V @ v
    return R, t

def compose(R1,t1,R2,t2):
    return R1@R2, R1@t2 + t1

def apply_inc(R,t,dxi):
    dR,dt = se3_exp(dxi)
    return compose(dR,dt,R,t)

def rot_err_deg(R_est,R_gt):
    dR = R_est @ R_gt.T
    c = np.clip((np.trace(dR)-1)*0.5, -1, 1)
    return float(np.degrees(np.arccos(c)))

# ---------------- camera ----------------
def make_K():
    return np.array([[500.,0,320.],[0,500.,240.],[0,0,1.]], dtype=float)

def transform(R,t,X):
    return (R @ X.T).T + t[None,:]

def project(K,Xc):
    z = np.where(np.abs(Xc[:,2:3])<1e-9, 1e-9, Xc[:,2:3])
    xn = Xc[:,:2]/z
    uv = np.empty((Xc.shape[0],2))
    uv[:,0] = K[0,0]*xn[:,0] + K[0,2]
    uv[:,1] = K[1,1]*xn[:,1] + K[1,2]
    return uv

def jacobian_2x6(K, Xc):
    fx,fy = K[0,0], K[1,1]
    x,y,z = Xc
    z2 = max(z*z, 1e-12)
    Jp = np.array([[fx/z,0,-fx*x/z2],[0,fy/z,-fy*y/z2]], dtype=float)
    dXdv = np.eye(3)
    dXdw = np.array([[0,z,-y],[-z,0,x],[y,-x,0]], dtype=float)
    return Jp @ np.hstack([dXdv,dXdw])

def residual_and_jac(K,R,t,X,uv_obs):
    Xc = transform(R,t,X)
    uv = project(K,Xc)
    r = (uv-uv_obs).reshape(-1)
    n = X.shape[0]
    J = np.zeros((2*n,6))
    for i in range(n):
        J[2*i:2*i+2,:] = jacobian_2x6(K,Xc[i])
    return r,J

# ---------------- weighting / info ----------------
def base_weights(C,sigma,mode,eps=1e-9):
    if mode=="conf":
        w = C.copy()
    elif mode=="unc":
        w = 1.0/(sigma**2 + eps)
    else:
        w = C/(sigma**2 + eps)  # conf_unc and IG variants
    return np.clip(w, eps, 1e9)

def Hi_list(J,w):
    out=[]
    for i in range(len(w)):
        Ji = J[2*i:2*i+2,:]
        out.append(Ji.T @ (w[i]*np.eye(2)) @ Ji)
    return out

def safe_logdet(A, reg=1e-9):
    Ar = A + reg*np.eye(A.shape[0])
    s,ld = np.linalg.slogdet(Ar)
    if s<=0:
        ev = np.linalg.eigvalsh(Ar)
        shift = max(0.0, -float(ev.min()) + reg)
        s,ld = np.linalg.slogdet(Ar + shift*np.eye(A.shape[0]))
    return float(ld)

def info_gain(Hi, lam=1e-6):
    H = np.sum(Hi, axis=0)
    ld_full = safe_logdet(H + lam*np.eye(6))
    ig = np.zeros(len(Hi))
    for i,h in enumerate(Hi):
        ld_wo = safe_logdet(H-h + lam*np.eye(6))
        ig[i] = max(0.0, ld_full-ld_wo)
    return ig, H

def ig_gate(ig, mode, eps=1e-9, T=0.5):
    ig = np.nan_to_num(ig, nan=0.0, posinf=0.0, neginf=0.0)
    if mode=="conf_unc_ig_ratio":
        g = ig/(ig.mean()+eps)
    elif mode=="conf_unc_ig_softmax":
        z = ig/max(T,eps)
        z = z - z.max()
        p = np.exp(z); p = p/(p.sum()+eps)
        g = p*len(ig)  # mean~1
    else:
        g = np.ones_like(ig)
    return np.clip(g, eps, 1e6)

# ---------------- solver ----------------
def solve_gn(K,X,uv_obs,w,R0,t0,max_iter=20,lam=1e-6):
    R,t = R0.copy(), t0.copy()
    prev = np.inf
    fail = False
    for _ in range(max_iter):
        r,J = residual_and_jac(K,R,t,X,uv_obs)
        W = np.repeat(w,2)
        sw = np.sqrt(np.clip(W,1e-12,1e12))
        rw = r*sw
        Jw = J*sw[:,None]
        H = Jw.T@Jw + lam*np.eye(6)
        g = Jw.T@rw
        try:
            dxi = -np.linalg.solve(H,g)
        except np.linalg.LinAlgError:
            fail=True; break
        Rn,tn = apply_inc(R,t,dxi)
        rn,_ = residual_and_jac(K,Rn,tn,X,uv_obs)
        c = 0.5*float(np.sum((rn*sw)**2))
        if c > prev + 1e-8:
            dxi *= 0.5
            Rn,tn = apply_inc(R,t,dxi)
            rn,_ = residual_and_jac(K,Rn,tn,X,uv_obs)
            c = 0.5*float(np.sum((rn*sw)**2))
        R,t = Rn,tn
        prev = c
        if np.linalg.norm(dxi)<1e-8: break
        if (not np.isfinite(c)) or c>1e12:
            fail=True; break
    return R,t,float(prev),fail

# ---------------- scenarios ----------------
def sample_normal(rng,n):
    X=np.empty((n,3))
    X[:,0]=rng.uniform(-2,2,n); X[:,1]=rng.uniform(-1.5,1.5,n); X[:,2]=rng.uniform(4,12,n)
    return X

def sample_planar(rng,n):
    X=np.empty((n,3))
    X[:,0]=rng.uniform(-3,3,n); X[:,1]=rng.uniform(-2,2,n); X[:,2]=8+rng.normal(0,0.05,n)
    return X

def pose_for(name):
    if name=="normal_3d": w,t=np.array([0.02,-0.01,0.015]),np.array([0.25,-0.05,0.1])
    elif name=="planar_scene": w,t=np.array([0.01,0,0.02]),np.array([0.2,0,0.05])
    elif name=="forward_motion": w,t=np.array([0.01,0,0]),np.array([0.02,0,0.5])
    elif name=="pure_rotation": w,t=np.array([0.08,-0.03,0.05]),np.array([0,0,0])
    elif name=="low_parallax": w,t=np.array([0.005,0,0.005]),np.array([0.005,0,0.04])
    elif name=="redundant_correspondences": w,t=np.array([0.01,0,0]),np.array([0.25,0,0.08])
    elif name=="outlier_contamination": w,t=np.array([0.02,-0.01,0.02]),np.array([0.2,-0.03,0.12])
    else: raise ValueError(name)
    return so3_exp(w), t

def gen_scene(name,seed,n,noise):
    rng=np.random.default_rng(seed)
    K=make_K()
    X = sample_planar(rng,n) if name=="planar_scene" else sample_normal(rng,n)
    if name=="redundant_correspondences":
        b = sample_normal(rng,max(10,n//6))
        reps=int(np.ceil(n/len(b)))
        X=np.vstack([b+rng.normal(0,0.01,b.shape) for _ in range(reps)])[:n]
    if name=="low_parallax":
        X[:,2]=rng.uniform(15,30,n)

    Rgt,tgt = pose_for(name)
    uv2 = project(K, transform(Rgt,tgt,X))
    uv_obs = uv2 + rng.normal(0,noise,uv2.shape)

    C = np.clip(rng.normal(0.85,0.08,n),0.05,1.0)
    sigma = np.clip(rng.lognormal(np.log(max(noise,0.3)),0.25,n),0.05,10.0)

    if name=="outlier_contamination":
        m=max(1,n//8); idx=rng.choice(n,m,replace=False)
        uv_obs[idx]+=rng.normal(0,25,(m,2))
        C[idx]=np.clip(C[idx]*0.5,0.01,1.0)
        sigma[idx]=np.clip(sigma[idx]*2,0.05,20.0)

    return dict(name=name,K=K,X=X,uv_obs=uv_obs,R_gt=Rgt,t_gt=tgt,C=C,sigma=sigma)

# ---------------- run ----------------
METHODS=["conf","unc","conf_unc","conf_unc_ig_ratio","conf_unc_ig_softmax"]
SCENARIOS=["normal_3d","planar_scene","forward_motion","pure_rotation","low_parallax","redundant_correspondences","outlier_contamination"]

def info_metrics(H):
    e=np.linalg.eigvalsh(0.5*(H+H.T))
    lam_min=float(e.min()); lam_max=float(e.max())
    kappa=float(lam_max/max(lam_min,1e-12))
    return lam_min,kappa

def run_method(scene,method,seed):
    rng=np.random.default_rng(seed+1000)
    R0 = so3_exp(np.array([0.02,-0.01,0.015])+rng.normal(0,0.01,3))
    t0 = np.array([0.05,-0.02,0.03])+rng.normal(0,0.02,3)

    _,J0 = residual_and_jac(scene["K"],R0,t0,scene["X"],scene["uv_obs"])
    base_mode = method if method in ["conf","unc","conf_unc"] else "conf_unc"
    wb = base_weights(scene["C"],scene["sigma"],base_mode)
    Hi = Hi_list(J0,wb)
    ig,H = info_gain(Hi)
    lam_min,kappa = info_metrics(H)

    if "ig_" in method:
        g = ig_gate(ig,method)
        w = np.clip(wb*g,1e-9,1e9)
    else:
        w = wb

    Rest,test,cost,fail = solve_gn(scene["K"],scene["X"],scene["uv_obs"],w,R0,t0)
    te = float(np.linalg.norm(test-scene["t_gt"]))
    re = rot_err_deg(Rest,scene["R_gt"])
    if (not np.isfinite(kappa)) or (kappa>1e10) or (not np.isfinite(cost)):
        fail=True
    return dict(
        scenario=scene["name"], method=method,
        translation_error=te, rotation_error_deg=re,
        final_residual=cost, lambda_min_H=lam_min, kappa_H=kappa, failure=int(fail)
    ), ig, wb, Hi

def ablation(scene,seed):
    R0 = so3_exp(np.array([0.01,0,0.01])); t0=np.array([0.02,0,0.02])
    _,J = residual_and_jac(scene["K"],R0,t0,scene["X"],scene["uv_obs"])
    wb = base_weights(scene["C"],scene["sigma"],"conf_unc")
    Hi = Hi_list(J,wb)
    ig,_ = info_gain(Hi)
    tr=np.array([np.trace(h) for h in Hi])

    C,s=scene["C"],scene["sigma"]
    best=None
    for i in range(len(C)):
        for j in range(i+1,len(C)):
            if abs(C[i]-C[j])<0.03 and abs(s[i]-s[j])<0.1:
                rr=(ig[i]+1e-9)/(ig[j]+1e-9); rr=max(rr,1/rr)
                if best is None or rr>best[0]: best=(rr,i,j)
    if best is None:
        i,j=int(np.argmax(ig)),int(np.argmin(ig))
    else:
        _,i,j=best
    return pd.DataFrame([
        {"id":i,"conf":C[i],"sigma":s[i],"trace_Hi":tr[i],"IG":ig[i]},
        {"id":j,"conf":C[j],"sigma":s[j],"trace_Hi":tr[j],"IG":ig[j]},
    ])

def plot_metrics(df,outdir):
    os.makedirs(outdir,exist_ok=True)
    for m in ["translation_error","rotation_error_deg","lambda_min_H","kappa_H"]:
        plt.figure(figsize=(10,4))
        piv=df.pivot(index="scenario",columns="method",values=m)
        piv.plot(kind="bar",ax=plt.gca())
        plt.title(m); plt.tight_layout()
        plt.savefig(os.path.join(outdir,f"{m}.png"),dpi=140); plt.close()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--num-points",type=int,default=120)
    ap.add_argument("--noise-px",type=float,default=1.0)
    ap.add_argument("--output-dir",type=str,default="results")
    args=ap.parse_args()

    os.makedirs(args.output_dir,exist_ok=True)
    rows=[]
    for si,sn in enumerate(SCENARIOS):
        print("[INFO] scenario:",sn)
        scene=gen_scene(sn,args.seed+10*si,args.num_points,args.noise_px)
        for mi,m in enumerate(METHODS):
            row,_,_,_ = run_method(scene,m,args.seed+100*si+mi)
            rows.append(row)

    df=pd.DataFrame(rows)
    df.to_csv(os.path.join(args.output_dir,"summary.csv"),index=False)
    with open(os.path.join(args.output_dir,"summary.md"),"w",encoding="utf-8") as f:
        f.write("# Summary Results\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n## Mean by method\n\n")
        f.write(df.groupby("method",as_index=False).mean(numeric_only=True).to_markdown(index=False))

    scene_ab = gen_scene("pure_rotation",args.seed+999,args.num_points,args.noise_px)
    ab = ablation(scene_ab,args.seed)
    ab.to_csv(os.path.join(args.output_dir,"ablation_equal_conf_sigma.csv"),index=False)
    with open(os.path.join(args.output_dir,"ablation_equal_conf_sigma.md"),"w",encoding="utf-8") as f:
        f.write("# Ablation: C~ , sigma~ but IG different\n\n")
        f.write(ab.to_markdown(index=False))

    plot_metrics(df, os.path.join(args.output_dir,"plots"))
    print("[DONE] outputs in:", args.output_dir)

if __name__=="__main__":
    main()
