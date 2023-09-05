def run_exp(cfg):

    # -- init --
    th.cuda.init()
    timer = ExpTimer()
    memer = GpuMemer()

    # -- read data --
    vid,fflow,bflow = load_sample(cfg)

    # -- acc flows --
    B,T,_,H,W = fflow.shape
    acc_flows = edict()
    acc_flows.fflow = th.zeros((B,T,T-1,2,H,W),device=fflow.device,dtype=fflow.dtype)
    acc_flows.bflow = th.zeros((B,T,T-1,2,H,W),device=fflow.device,dtype=fflow.dtype)


    # -- run search before for refine --
    dists,inds = None,None
    if cfg.search_name == "refine":
        dists,inds = run_nls(cfg,vid,fflow,bflow)

    # -- time forward --
    with th.no_grad():
        with TimeIt(timer,"fwd_nograd"):
            with MemIt(memer,"fwd_nograd"):
                run_method(cfg,cfg.search_name,vid,fflow,bflow,acc_flows,inds)

    # -- enable gradient --
    vid = vid.requires_grad_(True)

    # -- time forward --
    with TimeIt(timer,"fwd"):
        with MemIt(memer,"fwd"):
            outs = run_method(cfg,cfg.search_name,vid,fflow,bflow,acc_flows,inds)
    print("[%s] outs.shape: "%cfg.search_name,outs[0].shape)

    # -- time bwd --
    with TimeIt(timer,"bwd"):
        with MemIt(memer,"bwd"):
            loss = th.mean(outs[0])#th.mean([th.mean(o) for o in outs])
            loss.backward()

    # -- format results --
    results = edict()
    for key,val in timer.items():
        results[key] = val
    for key,(res,alloc) in memer.items():
        results["res_%s"%key] = res
        results["alloc_%s"%key] = alloc

    return results
