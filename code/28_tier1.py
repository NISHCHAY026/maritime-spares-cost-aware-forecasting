"""
Tier-1 strengthening, clean window (train Q01-16, test Q17-20).

#1 Scaled-error accuracy axis (MASE, RMSSE) -> re-derive per-SKU tau and
   check robustness vs the MAE-based tau (+0.39).
#2 Conformalize predictive quantiles (global multiplicative conformal on
   the training window) -> re-run fill-matched native-quantile.
#4 Cost-config (80-cell) + service-level (5-scenario) sweeps on the clean
   window: does deployed-dominance hold; is the cost-winner stable.
#5 ZIP fill-matched delta across sampling seeds -> is the -15% real.
#3 Simulator consistency check: simulated deployed mean on-hand vs the
   Stockmax 'In Stock' snapshot (true stockout-history validation is not
   possible — the data has no realized-stockout records).
"""
from __future__ import annotations
import time, warnings, json
from dataclasses import replace
from itertools import product
import numpy as np
import polars as pl
import config as C
from baselines import sba as sba_fn, ses as ses_fn
from models_classical import croston, tsb, adida, imapa
from models_distributional import fit_zip, sample_zip, fit_hurdle_nb, sample_hurdle_nb, quantiles_from_samples
from models_ml import LGBForecaster
from simulator import CostConfig, normal_policy, simulate, lead_days_to_qtrs, service_level_for
from policy_native_quantile import native_quantile_policy
from unified_stats import per_sku_kendall_tau, bootstrap_mean_ci

SD = C.SAMPLE_DIR; ROB = SD / "robustness"
AC = [f"ACTUALS_QTR{i:02d}" for i in range(1, 29)]
TRH, TLO, THI = 16, 16, 20
QS = (50, 80, 90, 95, 99)


def load():
    s = (pl.scan_parquet(SD/"sample_skus.parquet").select(
        ["STOCK_ITEM_NUMBER","FORECAST_ID","SEGMENTATION_GRP","ABC_GRP","STRATEGY_GRP",
         "VELOCITY_MODE","CRITICALITY_MODE","STOCKCLASS_MODE","UNIT_PRICE_USD",
         "LEAD_TIME_MEAN","USAGE_PER_YEAR_MEAN","SBA_ALPHA","SES_ALPHA"]).unique(
        subset=["STOCK_ITEM_NUMBER","FORECAST_ID"]).collect())
    p = (pl.scan_parquet(SD/"panel.parquet").select(["STOCK_ITEM_NUMBER","FORECAST_ID"]+AC)
         .unique(subset=["STOCK_ITEM_NUMBER","FORECAST_ID"]).collect())
    return s.join(p, on=["STOCK_ITEM_NUMBER","FORECAST_ID"], how="inner").sort(["STOCK_ITEM_NUMBER", "FORECAST_ID"])


def fit_models(df, train, H):
    sba_a = df["SBA_ALPHA"].cast(pl.Float64).fill_null(0.1).to_numpy()
    ses_a = df["SES_ALPHA"].cast(pl.Float64).fill_null(0.5).to_numpy()
    def enc(c):
        v = df[c].fill_null("__NA__").cast(pl.Utf8); cats = sorted(set(v.to_list()))
        m = {x:i for i,x in enumerate(cats)}; return np.array([m[x] for x in v.to_list()], float)
    F = {}
    F["SBA"]={"pt":sba_fn(train,sba_a).forecast[:,-1]}
    st=ses_fn(train,ses_a); F["SES"]={"pt":ses_a*train[:,-1]+(1-ses_a)*st[:,-1]}
    F["MA"]={"pt":(train[:,-1]+train[:,-2])/2}
    F["CROSTON"]={"pt":croston(train,sba_a,sba=False).forecast[:,-1]}
    F["TSB"]={"pt":tsb(train,sba_a).forecast[:,-1]}
    F["ADIDA"]={"pt":adida(train,ses_a,k=None)["forecast_per_period"]}
    F["IMAPA"]={"pt":imapa(train,ses_a,k_max=None)["forecast_per_period"]}
    zp=fit_zip(train); F["ZIP"]={"pt":zp["mean"],"zp":zp}
    hp=fit_hurdle_nb(train); F["HURDLE_NB"]={"pt":hp["mean"],"hp":hp}
    static={k:enc(c) for k,c in [("f_seg","SEGMENTATION_GRP"),("f_abc","ABC_GRP"),
            ("f_strategy","STRATEGY_GRP"),("f_velocity","VELOCITY_MODE"),
            ("f_criticality","CRITICALITY_MODE"),("f_stockclass","STOCKCLASS_MODE")]}
    static["f_price"]=np.where(df["UNIT_PRICE_USD"].fill_null(1.0).cast(pl.Float64).to_numpy()>0,
                               df["UNIT_PRICE_USD"].fill_null(1.0).cast(pl.Float64).to_numpy(),1.0)
    static["f_leadtime"]=df["LEAD_TIME_MEAN"].fill_null(60.0).cast(pl.Float64).to_numpy()
    static["f_usage"]=df["USAGE_PER_YEAR_MEAN"].fill_null(0.0).cast(pl.Float64).to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lg=LGBForecaster(lags=(1,2,3,4),rolls=(2,4)); lg.fit(train,static,train_quantiles=[q/100 for q in QS],seed=42)
        lp=lg.predict_recursive(train,static,h=H)
    F["LGBM"]={"pt":lp["mean"][:,0],"q":{q:np.maximum(lp["quantiles"].get(q/100,lp["mean"][:,0]),0) for q in QS}}
    # quantiles for ZIP/HNB at default seed
    F["ZIP"]["q"]={q:zq for q,zq in zip(QS,[quantiles_from_samples(sample_zip(zp["pi"],zp["lam"],1000,42),qs=[q/100 for q in QS])[q/100] for q in QS])}
    F["HURDLE_NB"]["q"]={q:quantiles_from_samples(sample_hurdle_nb(hp["p"],hp["mu"],hp["alpha"],1000,42),qs=[q/100 for q in QS])[q/100] for q in QS}
    cp=SD/"forecasts_chronos_clean.parquet"
    if cp.exists():
        cd=df.select(["STOCK_ITEM_NUMBER","FORECAST_ID"]).join(pl.read_parquet(cp),on=["STOCK_ITEM_NUMBER","FORECAST_ID"],how="left")
        F["CHRONOS"]={"pt":cd["POINT_Q17"].fill_null(0.0).to_numpy(),"q":{q:cd[f"Q_{q:02d}"].fill_null(0.0).to_numpy() for q in QS}}
    return F


def main():
    t0=time.time(); cfg=CostConfig()
    df=load(); N=df.height
    A=df.select(AC).to_numpy().astype(float); train=A[:,:TRH]; test=A[:,TLO:THI]; H=test.shape[1]
    pos=test.sum(1)>0
    price=np.where(df["UNIT_PRICE_USD"].fill_null(1.0).cast(pl.Float64).to_numpy()>0,df["UNIT_PRICE_USD"].fill_null(1.0).cast(pl.Float64).to_numpy(),1.0)
    ld=df["LEAD_TIME_MEAN"].fill_null(60.0).cast(pl.Float64).to_numpy(); ld=np.where(ld>0,ld,60.0)
    lq=lead_days_to_qtrs(ld); lsq=(ld*0.3)/91.3125; sigma=train.std(1,ddof=0)
    sl=np.array([service_level_for(c,cfg) for c in df["CRITICALITY_MODE"].fill_null("Normal").to_list()])
    F=fit_models(df,train,H); models=list(F)
    def simcost(pt,c=cfg):
        pol=normal_policy(np.maximum(pt,0),sigma,lq.astype(float),lsq,price,sl,c)
        return simulate(test,pol["s"],pol["S"],lq,price,cfg=c)
    cost=np.zeros((N,len(models)))
    for j,m in enumerate(models):
        cost[:,j]=simcost(F[m]["pt"])["total_cost"]
    mae=np.zeros((N,len(models)))
    for j,m in enumerate(models):
        mae[:,j]=np.abs(np.repeat(np.maximum(F[m]["pt"],0)[:,None],H,1)-test).mean(1)

    out=["TIER-1 (clean window train Q01-16 / test Q17-20)","="*64,f"N={N:,} demand-active={int(pos.sum()):,}",""]

    # ---- #1 MASE / RMSSE ----
    d=np.abs(np.diff(train,axis=1)).mean(1)        # naive-1 in-sample MAE per SKU
    valid=d>0
    mase=np.full_like(mae,np.nan)
    mase[valid]=mae[valid]/d[valid,None]
    # RMSSE
    rmsse_den=np.sqrt((np.diff(train,axis=1)**2).mean(1)); rv=rmsse_den>0
    rmse=np.sqrt(((np.repeat(0,1),))) if False else None
    rmsse=np.full_like(mae,np.nan)
    for j,m in enumerate(models):
        se=((np.repeat(np.maximum(F[m]["pt"],0)[:,None],H,1)-test)**2).mean(1)
        rmsse[rv,j]=np.sqrt(se[rv])/rmsse_den[rv]
    # tau on demand-active & valid scaling
    pv=pos & valid
    tau_mae=bootstrap_mean_ci(per_sku_kendall_tau(mae[pv],cost[pv]))
    tau_mase=bootstrap_mean_ci(per_sku_kendall_tau(mase[pv],cost[pv]))
    pr=pos & rv
    tau_rmsse=bootstrap_mean_ci(per_sku_kendall_tau(rmsse[pr],cost[pr]))
    out+=["#1 SCALED-ERROR ROBUSTNESS (demand-active SKUs):",
          f"  tau(MAE-rank , cost-rank)   = {tau_mae['mean']:+.3f} [{tau_mae['lo']:+.3f},{tau_mae['hi']:+.3f}]  N={tau_mae['n_valid']:,}",
          f"  tau(MASE-rank, cost-rank)   = {tau_mase['mean']:+.3f} [{tau_mase['lo']:+.3f},{tau_mase['hi']:+.3f}]  N={tau_mase['n_valid']:,}",
          f"  tau(RMSSE-rank, cost-rank)  = {tau_rmsse['mean']:+.3f} [{tau_rmsse['lo']:+.3f},{tau_rmsse['hi']:+.3f}]  N={tau_rmsse['n_valid']:,}",""]

    # ---- helpers for native fill-matched ----
    def fillmatched(qd, pt):
        poln=normal_policy(np.maximum(pt,0),sigma,lq.astype(float),lsq,price,sl,cfg)
        tgt=simulate(test,poln["s"],poln["S"],lq,price,cfg=cfg)["fill_rate"].mean()
        cn=simulate(test,poln["s"],poln["S"],lq,price,cfg=cfg)["total_cost"].mean()
        qarr={0.5:qd[50],0.8:qd[80],0.9:qd[90],0.95:qd[95],0.99:qd[99]}
        polv=native_quantile_policy(qarr,np.maximum(pt,0),lq.astype(float),price,sl,cfg)
        sb,eoq=polv["s"],polv["eoq"]; lo,hi=0.0,8.0
        cap=simulate(test,hi*sb,hi*sb+eoq,lq,price,cfg=cfg)["fill_rate"].mean()<tgt
        if not cap:
            for _ in range(38):
                mid=(lo+hi)/2
                if simulate(test,mid*sb,mid*sb+eoq,lq,price,cfg=cfg)["fill_rate"].mean()<tgt: lo=mid
                else: hi=mid
            k=(lo+hi)/2
        else: k=hi
        cm=simulate(test,k*sb,k*sb+eoq,lq,price,cfg=cfg)["total_cost"].mean()
        return k,cap,100*(cm-cn)/cn

    # ---- #5 ZIP seed stress ----
    zp=F["ZIP"]["zp"]; deltas=[]
    for sd in (42,1,2,3,4):
        zq={q:quantiles_from_samples(sample_zip(zp["pi"],zp["lam"],1000,sd),qs=[q/100 for q in QS])[q/100] for q in QS}
        _,_,dd=fillmatched(zq,F["ZIP"]["pt"]); deltas.append(dd)
    out+=["#5 ZIP fill-matched cost delta across sampling seeds:",
          f"  seeds 42,1,2,3,4 -> {['%+.1f%%'%x for x in deltas]}",
          f"  mean {np.mean(deltas):+.1f}%  range [{min(deltas):+.1f}%, {max(deltas):+.1f}%]",""]

    # ---- #2 conformal (global multiplicative on training) ----
    out+=["#2 CONFORMALIZED native-quantile (global mult. conformal on train):",
          f"  {'MODEL':<10} {'rawΔ%':>8} {'confΔ%':>8} {'raw cap':>8} {'conf cap':>8}"]
    qmods=[m for m in models if "q" in F[m]]
    conf_summary={}
    for m in qmods:
        raw_k,raw_cap,raw_d=fillmatched(F[m]["q"],F[m]["pt"])
        # conformal factors per level from training pooled coverage
        cq={}
        for q in QS:
            Q=np.maximum(F[m]["q"][q],1e-9)
            ratio=(train/Q[:,None]).ravel()
            ratio=ratio[np.isfinite(ratio)]
            c=float(np.quantile(ratio,q/100))
            cq[q]=np.maximum(F[m]["q"][q]*c,0)
        ck,ccap,cd=fillmatched(cq,F[m]["pt"])
        conf_summary[m]={"raw_delta":raw_d,"conf_delta":cd,"raw_capped":raw_cap,"conf_capped":ccap}
        out.append(f"  {m:<10} {raw_d:>7.1f}% {cd:>7.1f}% {str(raw_cap):>8} {str(ccap):>8}")
    out.append("")

    # ---- #4 sweeps ----
    pp=(pl.scan_parquet(SD/"policy_panel.parquet").select(["STOCK_ITEM_NUMBER","FORECAST_ID","DEPLOYED_NEW_MIN_MEAN","DEPLOYED_NEW_MAX_MEAN"]).unique(subset=["STOCK_ITEM_NUMBER","FORECAST_ID"]).collect())
    ov=df.select(["STOCK_ITEM_NUMBER","FORECAST_ID"]).with_row_index("idx").join(pp,on=["STOCK_ITEM_NUMBER","FORECAST_ID"],how="inner")
    idx=ov["idx"].to_numpy()
    sdep=ov["DEPLOYED_NEW_MIN_MEAN"].fill_null(0.0).cast(pl.Float64).to_numpy()
    Sdep=np.maximum(ov["DEPLOYED_NEW_MAX_MEAN"].fill_null(0.0).cast(pl.Float64).to_numpy(),sdep+1.0)
    grid=list(product([0.15,0.20,0.25,0.30,0.35],[25.,50.,100.,250.],[50.,100.,250.,500.]))
    dep_dom=0; winners={}
    for h,k,so in grid:
        c=replace(cfg,holding_rate_annual=h,ordering_cost=k,stockout_unit_cost=so)
        cm={m:simcost(F[m]["pt"],c)["total_cost"][idx].mean() for m in models}
        depc=simulate(test[idx],sdep,Sdep,lq[idx],price[idx],cfg=c)["total_cost"].mean()
        best=min(cm,key=cm.get); winners[best]=winners.get(best,0)+1
        if cm[best]<depc: dep_dom+=1
    out+=["#4 COST-CONFIG SWEEP (80 cells) on clean window:",
          f"  deployed beaten by best model in {dep_dom}/80 cells",
          f"  cost-winner identity: "+", ".join(f"{k}:{v}" for k,v in sorted(winners.items(),key=lambda x:-x[1]))]
    SLS={"default":(0.99,0.97,0.95),"conservative":(0.99,0.99,0.99),"aggressive":(0.90,0.90,0.90),"flat95":(0.95,0.95,0.95)}
    sl_rows=[]
    for nm,(cr,no,lo_) in SLS.items():
        c=replace(cfg,sl_critical=cr,sl_normal=no,sl_low=lo_)
        slv=np.array([service_level_for(x,c) for x in df["CRITICALITY_MODE"].fill_null("Normal").to_list()])
        def sc(pt):
            pol=normal_policy(np.maximum(pt,0),sigma,lq.astype(float),lsq,price,slv,c)
            return simulate(test,pol["s"],pol["S"],lq,price,cfg=c)["total_cost"]
        cm={m:sc(F[m]["pt"])[idx].mean() for m in models}
        depc=simulate(test[idx],sdep,Sdep,lq[idx],price[idx],cfg=c)["total_cost"].mean()
        best=min(cm,key=cm.get)
        sl_rows.append(f"  {nm:<12} best={best:<10} ${cm[best]:,.0f}  deployed ${depc:,.0f}  gap {100*(cm[best]-depc)/depc:+.0f}%")
    out+=["", "#4 SERVICE-LEVEL SWEEP on clean window:"]+sl_rows+[""]

    # ---- #3 simulator consistency vs In Stock ----
    sm=(pl.scan_parquet(C.OUT_FILES["stockmax"]).select(["Parttypeno","In Stock"]).rename({"Parttypeno":"STOCK_ITEM_NUMBER"})
        .group_by("STOCK_ITEM_NUMBER").agg(pl.col("In Stock").cast(pl.Float64,strict=False).mean().alias("instock")).collect())
    ovkeys=df.select(["STOCK_ITEM_NUMBER"]).with_row_index("ridx")
    j=ovkeys.join(sm,on="STOCK_ITEM_NUMBER",how="inner")
    # simulated mean on-hand under deployed for overlap
    simdep=simulate(test[idx],sdep,Sdep,lq[idx],price[idx],cfg=cfg)
    moh=simdep["mean_on_hand"]
    ov_is=ov.join(sm,on="STOCK_ITEM_NUMBER",how="left")["instock"].fill_null(0.0).to_numpy()
    mask=ov_is>0
    if mask.sum()>10:
        corr=float(np.corrcoef(np.log1p(moh[mask]),np.log1p(ov_is[mask]))[0,1])
        ratio=float(np.median(moh[mask]/np.maximum(ov_is[mask],1e-9)))
    else:
        corr,ratio=float("nan"),float("nan")
    out+=["#3 SIMULATOR CONSISTENCY (no realized-stockout history exists; partial check):",
          f"  simulated deployed mean on-hand vs recorded In Stock (overlap, instock>0, n={int(mask.sum()):,}):",
          f"    log-log corr = {corr:+.3f}   median(sim/recorded) = {ratio:.2f}",
          f"  simulated deployed fill = {simdep['fill_rate'].mean():.3f}  (operator's tiered availability goal 0.95-0.99)",
          "  -> true validation requires realized fill/stockout records, which the dataset lacks."]

    rep="\n".join(out)
    (ROB/"tier1.txt").write_text(rep,encoding="utf-8")
    (ROB/"tier1.json").write_text(json.dumps({
        "tau_mae":tau_mae,"tau_mase":tau_mase,"tau_rmsse":tau_rmsse,
        "zip_seed_deltas":deltas,"conformal":conf_summary,
        "dep_dom_cells":dep_dom,"winners":winners,
        "sim_corr":corr,"sim_ratio":ratio,"sim_dep_fill":float(simdep['fill_rate'].mean())},indent=2,default=lambda o:bool(o) if isinstance(o,np.bool_) else (float(o) if isinstance(o,(np.floating,np.integer)) else str(o))),encoding="utf-8")
    print(rep); print(f"\nDone {time.time()-t0:.0f}s")


if __name__=="__main__":
    main()
